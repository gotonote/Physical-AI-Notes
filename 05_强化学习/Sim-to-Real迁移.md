# Sim-to-Real 迁移

Sim-to-Real（仿真到真实）迁移是机器人学习中的关键技术，解决在仿真环境中训练的策略如何有效迁移到真实机器人的问题。

## 目录

- [1. Sim-to-Real概述](#1-sim-to-real概述)
- [2. 领域随机化](#2-领域随机化)
- [3. 域适应](#3-域适应)
- [4. 课程学习](#4-课程学习)
- [5. 系统识别](#5-系统识别)
- [6. 实践策略](#6-实践策略)

---

## 1. Sim-to-Real概述

### 1.1 为什么需要Sim-to-Real

| 方面 | 仿真 | 真实 |
|------|------|------|
| 样本收集 | 快速、并行 | 慢、成本高 |
| 安全性 | 无风险 | 有风险 |
| 可重复性 | 高 | 低 |
| 物理真实性 | 不完美 | 真实 |
| 传感器噪声 | 简化 | 复杂 |

### 1.2 Sim-to-Real Gap

```
Sim-to-Real Gap = |性能(真实) - 性能(仿真)|

主要来源:
1. 视觉差异（纹理、光照、噪声）
2. 动力学差异（摩擦、质量、延迟）
3. 传感器差异（噪声、分辨率）
4. 动作执行差异（延迟、精度）
```

---

## 2. 领域随机化

### 2.1 视觉随机化

```python
import numpy as np
import cv2

class VisualDomainRandomization:
    """
    视觉领域随机化
    在仿真中随机化视觉参数以提高泛化能力
    """
    def __init__(self):
        pass
    
    def randomize_textures(self, image, texture_type='random'):
        """纹理随机化"""
        if texture_type == 'random':
            # 随机颜色抖动
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            
            # 随机改变亮度
            hsv[:, :, 2] *= np.random.uniform(0.5, 1.5)
            
            # 随机改变色调
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-20, 20)) % 180
            
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        elif texture_type == 'noise':
            # 添加噪声
            noise = np.random.normal(0, 10, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def randomize_camera(self, image):
        """相机参数随机化"""
        # 随机模糊
        if np.random.random() < 0.3:
            kernel_size = np.random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # 随机分辨率变化（模拟）
        # 随机畸变
        if np.random.random() < 0.3:
            h, w = image.shape[:2]
            K = np.array([[w/2, 0, w/2], [0, h/2, h/2], [0, 0, 1]], dtype=np.float32)
            
            # 随机畸变系数
            k1 = np.random.uniform(-0.1, 0.1)
            k2 = np.random.uniform(-0.1, 0.1)
            
            # 去畸变
            new_K = K.copy()
            new_K[0, 0] *= np.random.uniform(0.9, 1.1)
            new_K[1, 1] *= np.random.uniform(0.9, 1.1)
            
        return image
    
    def apply_randomization(self, image):
        """应用所有随机化"""
        image = self.randomize_textures(image, np.random.choice(['random', 'noise']))
        image = self.randomize_camera(image)
        return image
```

### 2.2 物理随机化

```python
class PhysicsDomainRandomization:
    """
    物理领域随机化
    随机化物理参数以提高对未建模动态的鲁棒性
    """
    def __init__(self):
        # 可随机化的物理参数
        self.params = {
            'friction': [0.1, 2.0],           # 摩擦系数
            'mass': [0.5, 2.0],                # 质量比例
            'restitution': [0.0, 0.5],         # 弹性系数
            'gravity': [9.6, 10.0],            # 重力加速度
            'motor_torque': [0.8, 1.2],        # 电机扭矩
            'latency': [0.0, 0.05],           # 延迟(秒)
            'joint_damping': [0.0, 1.0],      # 关节阻尼
        }
        
    def sample_params(self):
        """采样物理参数"""
        sampled = {}
        for param, (low, high) in self.params.items():
            if param == 'gravity':
                sampled[param] = np.random.uniform(low, high)
            else:
                sampled[param] = np.random.uniform(low, high)
        return sampled
    
    def apply_to_sim(self, sim, params):
        """将随机化参数应用到仿真"""
        # 设置摩擦
        sim.set_body_friction('robot_body', params['friction'])
        
        # 设置质量
        sim.set_body_mass('robot_body', params['mass'])
        
        # 设置重力
        sim.set_gravity(params['gravity'])
        
        # 设置延迟
        sim.set_action_latency(params['latency'])
        
        # ... 更多参数
```

### 2.3 自动领域随机化

```python
class AdaptiveDomainRandomization:
    """
    自动领域随机化 (ADR)
    根据真实环境性能自动调整随机化程度
    """
    def __init__(self, base_ranges):
        self.base_ranges = base_ranges  # 基础随机化范围
        self.current_ranges = base_ranges.copy()
        self.performance_history = []
        
    def update_ranges(self, real_performance):
        """根据真实性能更新随机化范围"""
        self.performance_history.append(real_performance)
        
        if len(self.performance_history) < 10:
            return
            
        # 检查性能趋势
        recent_perf = np.mean(self.performance_history[-5:])
        
        if recent_perf > threshold:
            # 性能良好，增加随机化难度
            self.increase_difficulty()
        else:
            # 性能下降，减小难度
            self.decrease_difficulty()
            
    def increase_difficulty(self):
        """增加随机化难度"""
        for param, (low, high) in self.current_ranges.items():
            # 扩大范围
            center = (low + high) / 2
            width = (high - low) * 1.2
            self.current_ranges[param] = (
                center - width / 2,
                center + width / 2
            )
```

---

## 3. 域适应

### 3.1 像素级域适应

```python
import torch
import torch.nn as nn

class PixelLevelDomainAdaptation:
    """
    像素级域适应
    将真实图像风格转换为仿真图像风格
    """
    def __init__(self):
        # 生成器
        self.generator = ResNetGenerator()
        
        # 判别器
        self.discriminator = PatchGANDiscriminator()
        
    def forward(self, real_image):
        # 生成仿真风格图像
        fake_image = self.generator(real_image)
        return fake_image
    
    def train_step(self, real_images, sim_images):
        # 重建损失
        reconstruction_loss = nn.L1Loss()(real_images, sim_images)
        
        # 对抗损失
        fake_pred = self.discriminator(real_images)
        real_loss = nn.BCEWithLogitsLoss()(fake_pred, torch.ones_like(fake_pred))
        
        return reconstruction_loss + 0.1 * real_loss
```

### 3.2 特征级域适应

```python
class FeatureLevelDomainAdaptation:
    """
    特征级域适应
    在特征空间中消除域差异
    """
    def __init__(self, feature_dim=256):
        # 特征提取器
        self.encoder = Encoder(feature_dim)
        
        # 域判别器
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 源/目标域分类
        )
        
        # 梯度反转层
        self.grl = GradientReversalLayer()
        
    def forward(self, source_features, target_features):
        # 源域特征
        source_pred = self.domain_classifier(self.grl(source_features))
        
        # 目标域特征
        target_pred = self.domain_classifier(self.grl(target_features))
        
        return source_pred, target_pred
    
    def domain_loss(self, source_pred, target_pred):
        """域适应损失"""
        source_loss = nn.CrossEntropyLoss()(source_pred, torch.zeros(len(source_pred)))
        target_loss = nn.CrossEntropyLoss()(target_pred, torch.ones(len(target_pred)))
        return source_loss + target_loss


class GradientReversalLayer(nn.Module):
    """梯度反转层"""
    def forward(self, x):
        return x
    
    def backward(self, grad):
        return -grad  # 反转梯度
```

---

## 4. 课程学习

### 4.1 渐进式课程学习

```python
class ProgressiveCurriculum:
    """
    渐进式课程学习
    从简单到复杂的任务逐渐增加难度
    """
    def __init__(self, task_generator):
        self.task_generator = task_generator
        self.current_difficulty = 0.0
        self.difficulty_schedule = 'linear'  # linear, exponential
        
    def get_current_task(self):
        """获取当前难度的任务"""
        return self.task_generator.generate(difficulty=self.current_difficulty)
    
    def update_difficulty(self, success_rate):
        """根据成功率更新难度"""
        if success_rate > 0.8:
            # 任务太简单，增加难度
            self.current_difficulty = min(1.0, self.current_difficulty + 0.05)
        elif success_rate < 0.3:
            # 任务太难，降低难度
            self.current_difficulty = max(0.0, self.current_difficulty - 0.1)
            
        return self.current_difficulty
    
    def train_with_curriculum(self, agent, num_iterations):
        """课程学习训练"""
        for iteration in range(num_iterations):
            task = self.get_current_task()
            
            # 训练
            performance = agent.train_on_task(task)
            
            # 更新难度
            self.update_difficulty(performance['success_rate'])
            
            # 定期在真实环境测试
            if iteration % 1000 == 0:
                real_performance = agent.evaluate_on_real()
                print(f"Iteration {iteration}, Real perf: {real_performance}")
```

---

## 5. 系统识别

### 5.1 在线系统识别

```python
class OnlineSystemIdentification:
    """
    在线系统识别
    在真实环境中在线估计物理参数
    """
    def __init__(self, param_bounds):
        self.param_bounds = param_bounds
        
        # 参数估计器
        self.param_estimator = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, len(param_bounds))
        )
        
    def estimate_params(self, states, actions, next_states):
        """
        从轨迹数据估计系统参数
        使用神经网络预测残差
        """
        # 构建输入
        x = torch.cat([states, actions], dim=-1)
        
        # 预测下一状态的变化
        delta_s = next_states - states
        
        # 估计物理参数
        params = self.param_estimator(x)
        
        return params
    
    def update_sim_with_estimated_params(self, sim, estimated_params):
        """使用估计的参数更新仿真"""
        for param_name, param_value in zip(self.param_bounds.keys(), estimated_params):
            sim.set_param(param_name, param_value)
```

### 5.2 Bayesian系统识别

```python
class BayesianSystemID:
    """
    Bayesian系统识别
    使用贝叶斯方法估计参数分布
    """
    def __init__(self, param_names, prior_means, prior_stds):
        self.param_names = param_names
        
        # 初始先验
        self.params = {
            name: {'mean': mean, 'std': std}
            for name, mean, std in zip(param_names, prior_means, prior_stds)
        }
        
    def update(self, observation, action, next_observation):
        """使用观测更新参数后验"""
        # 简化的贝叶斯更新
        # 实际中需要更复杂的粒子滤波器或UKF
        
        predicted_next = self.predict(observation, action)
        
        # 计算预测误差
        error = next_observation - predicted_next
        
        # 更新参数均值和方差
        for name in self.param_names:
            learning_rate = 0.01
            self.params[name]['mean'] += learning_rate * error
            self.params[name]['std'] *= 0.99  # 方差衰减
```

---

## 6. 实践策略

### 6.1 Sim-to-Real检查清单

```python
def sim_to_real_checklist():
    """
    Sim-to-Real 实践检查清单
    """
    checklist = {
        # 1. 仿真真实性
        'physics_accuracy': [
            "检查摩擦模型",
            "检查质量分布",
            "检查关节限制",
            "检查传感器延迟",
            "检查执行器动力学"
        ],
        
        # 2. 视觉真实性
        'visual_accuracy': [
            "检查纹理质量",
            "检查光照模型",
            "检查相机参数",
            "检查噪声模型"
        ],
        
        # 3. 域随机化
        'domain_randomization': [
            "视觉随机化",
            "物理随机化",
            "传感器噪声随机化"
        ],
        
        # 4. 训练策略
        'training_strategy': [
            "课程学习",
            "渐进式迁移",
            "在真实环境微调"
        ],
        
        # 5. 评估
        'evaluation': [
            "仿真性能基准",
            "真实性能评估",
            "失败模式分析"
        ]
    }
    
    return checklist
```

### 6.2 成功案例：机器人抓取

```python
class Sim2RealGrasping:
    """
    Sim-to-Real 抓取案例
    使用领域随机化成功迁移到真实机器人
    """
    def __init__(self):
        # 1. 仿真环境配置
        self.sim_config = {
            'physics': {
                'friction_range': (0.1, 1.5),
                'mass_range': (0.5, 2.0),
            },
            'vision': {
                'texture_variations': 1000,
                'lighting_variations': 50,
                'camera_noise': True
            }
        }
        
        # 2. 训练策略
        self.training_strategy = {
            'total_steps': 1000000,
            'domain_randomization': True,
            'curriculum': True
        }
        
    def train(self):
        """训练流程"""
        # Phase 1: 仿真训练（带领域随机化）
        # Phase 2: 真实环境评估
        # Phase 3: 失败案例分析
        # Phase 4: 调整随机化参数
        # Phase 5: 继续训练
        # Phase 6: 最终部署
        pass
```

---

## 参考文献

1. Tobin, J., et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS.
2. Peng, X. B., et al. (2018). Sim-to-Real Transfer with Domain Adaptation. arXiv.
3. Sadeghi, F., & Levine, S. (2017). CAD2RL: Real Single-Flight Flight Learning. RSS.

---

*本章节持续更新中...*
