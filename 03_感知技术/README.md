# 02 感知技术

感知技术是物理AI系统的核心组成部分，使机器人能够理解和交互于物理世界。本章将详细介绍各类感知技术原理、实现方法及在机器人系统中的应用。

## 目录

- [1. 感知技术概述](#1-感知技术概述)
  - [1.1 视觉感知](#11-视觉感知)
  - [1.2 触觉感知](#12-触觉感知)
  - [1.3 力控感知](#13-力控感知)
  - [1.4 IMU惯性测量](#14-imu惯性测量)
  - [1.5 其他传感器](#15-其他传感器)
- [2. 多模态感知融合](#2-多模态感知融合)
- [3. SLAM与定位技术](#3-slam与定位技术)
- [4. 实时目标检测与跟踪](#4-实时目标检测与跟踪)
- [5. 深度学习在感知中的应用](#5-深度学习在感知中的应用)
- [6. 传感器融合算法](#6-传感器融合算法)

---

## 1. 感知技术概述

物理AI系统的感知层负责从物理世界采集多模态信息，为决策和控制系统提供环境理解。

### 1.1 视觉感知

视觉是机器人感知环境的最重要方式之一。

#### 核心设备类型

| 设备类型 | 特点 | 应用场景 |
|---------|------|---------|
| 单目相机 | 成本低，信息丰富 | 目标检测，语义分割 |
| 双目相机 | 可获取深度信息 | 导航避障，3D重建 |
| RGB-D相机 | 实时深度图 | 室内导航，物体抓取 |
| 事件相机 | 高动态范围，低延迟 | 高速跟踪，快速响应 |

#### 代码示例：OpenCV相机标定

```python
import cv2
import numpy as np

class VisualPerception:
    def __init__(self, camera_type='rgb_d'):
        self.camera_type = camera_type
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def calibrate_camera(self, calibration_images):
        object_points = []
        image_points = []
        objp = np.zeros((9*6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                object_points.append(objp)
                image_points.append(corners)
        
        ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points, image_points, gray.shape[::-1], None, None
        )
        return self.camera_matrix, self.dist_coeffs
```

### 1.2 触觉感知

触觉传感器使机器人能够感知接触力、纹理和物体属性。

#### 触觉传感器类型

| 类型 | 原理 | 特点 | 典型应用 |
|-----|------|------|---------|
| 电阻式 | 电阻随压力变化 | 成本低，耐用 | 工业夹爪 |
| 电容式 | 电容变化检测压力 | 高灵敏度 | 精细操作 |
| 光纤式 | 光强随变形变化 | 抗电磁干扰 | 医疗机器人 |
| 压电式 | 压电效应 | 动态响应快 | 力反馈控制 |

### 1.3 力控感知

力控感知使机器人能够实现精确的力跟随和阻抗控制。

```python
class ImpedanceController:
    def __init__(self, M=5.0, B=50.0, K=100.0):
        self.M = M  # 惯性
        self.B = B  # 阻尼
        self.K = K  # 刚度
        
    def compute_force(self, desired_pos, actual_pos, desired_vel, actual_vel, ext_force=0):
        position_error = desired_pos - actual_pos
        velocity_error = desired_vel - actual_vel
        return self.K * position_error + self.B * velocity_error - ext_force
```

### 1.4 IMU惯性测量

IMU提供机器人姿态和运动状态的精确测量。

| 组件 | 测量内容 |
|------|---------|
| 加速度计 | 线性加速度 |
| 陀螺仪 | 角速度 |
| 磁力计 | 航向（可选） |

### 1.5 其他传感器

| 类型 | 原理 | 范围 | 特点 |
|-----|------|------|------|
| 超声波 | 飞行时间 | 0.1-10m | 成本低 |
| 红外 | 三角测量 | 0.1-4m | 精度高 |
| LiDAR | 飞行时间 | 0.1-200m | 3D点云 |
| 毫米波雷达 | 调频连续波 | 0.1-300m | 全天候 |

---

## 2. 多模态感知融合

多模态融合将不同传感器的信息整合，提供更鲁棒和准确的环境感知。

### 融合架构

- **早期融合**：原始数据/特征级融合
- **晚期融合**：决策级融合
- **注意力融合**：基于注意力机制的动态加权融合

---

## 3. SLAM与定位技术

SLAM使机器人在未知环境中同时实现定位和建图。

### 3.1 视觉SLAM

- 前端：特征提取、特征匹配、里程计
- 后端：BA优化、回环检测、全局优化

### 3.2 LiDAR SLAM

- 点云配准（ICP、NDT）
- 因子图优化

### 3.3 紧耦合SLAM

- 视觉-惯性融合 (VIO)
- LiDAR-惯性融合 (LIO)

---

## 4. 实时目标检测与跟踪

### 4.1 2D目标检测

- YOLO系列
- Faster R-CNN

### 4.2 3D目标检测

- PointPillars
- CenterPoint

### 4.3 多目标跟踪

- SORT
- DeepSORT

---

## 5. 深度学习在感知中的应用

### 5.1 CNN图像处理

- ResNet、VGG等骨干网络

### 5.2 Transformer视觉模型

- ViT、DETR

### 5.3 端到端感知模型

- BEVFormer
- TransFusion

---

## 6. 传感器融合算法

### 6.1 卡尔曼滤波

```python
class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # 状态转移矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 观测噪声协方差
        self.x = None  # 状态
        self.P = None  # 协方差
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(K)) - K @ self.H) @ self.P
        return self.x
```

### 6.2 粒子滤波

### 6.3 图优化

- G2O、iSAM2

---

*本章节持续更新中...*
