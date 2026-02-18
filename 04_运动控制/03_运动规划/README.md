# 03 运动规划

运动规划（Motion Planning）是机器人学的核心问题之一，旨在为机器人找到一条从起始状态到目标状态的无碰撞路径。

## 📋 目录

- [概述](#概述)
- [核心概念](#核心概念)
- [章节内容](#章节内容)
- [学习路径](#学习路径)

---

## 概述

```
┌─────────────────────────────────────────────────────────────┐
│                    运动规划问题定义                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入：                                                      │
│  • 机器人模型（运动学/动力学约束）                            │
│  • 起始状态 q_start                                         │
│  • 目标状态 q_goal                                          │
│  • 环境模型（障碍物）                                        │
│                                                             │
│  输出：                                                      │
│  • 无碰撞路径 π: [0,1] → Configuration Space               │
│  • 时间参数化的轨迹 τ: [0,T] → State Space                 │
│                                                             │
│  约束：                                                      │
│  • 避免碰撞（几何/运动学/动力学）                            │
│  • 满足运动学限制（关节限位）                                │
│  • 满足动力学限制（速度/加速度）                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心概念

### Configuration Space（构型空间）

**构型空间**是描述机器人所有可能位置的空间。

```
┌─────────────────────────────────────────────────────────────┐
│                 工作空间 vs 构型空间                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  工作空间（Workspace）：                                     │
│  • 末端执行器在笛卡尔空间中的位置和姿态                      │
│  • 维度：3D位置 + 3D姿态 = 6 DOF                            │
│  • 直观但计算复杂                                            │
│                                                             │
│  构型空间（Configuration Space, C-Space）：                 │
│  • 机器人所有关节角度组成的空间                              │
│  • 维度 = 关节数量                                          │
│  • 例：6-DOF机械臂 → 6维C-Space                             │
│  • 障碍物映射到C-Space → C_obs                              │
│                                                             │
│  优势：在C-Space中，机器人简化为一个点                       │
│                                                             │
│         工作空间                构型空间                     │
│    ┌──────────────┐         ┌──────────────┐               │
│    │   ╔═══╗      │         │    ******    │               │
│    │   ║机器人║   │   ──▶   │   *      *   │               │
│    │   ╚═══╝障碍  │         │  *   ·   *   │ ← 机器人是一个点│
│    │              │         │   *      *   │               │
│    └──────────────┘         │    ******    │ ← C_obs       │
│                             └──────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 路径 vs 轨迹

| 概念 | 定义 | 特点 |
|------|------|------|
| **路径（Path）** | 空间中的曲线，无时间信息 | 纯几何描述 |
| **轨迹（Trajectory）** | 带时间参数化的路径 | 包含速度、加速度 |

---

## 章节内容

### [路径规划](./路径规划.md)
- A* 算法
- RRT（快速探索随机树）
- RRT*（最优RRT）
- 势场法

### [运动学](./运动学.md)
- 正运动学（Forward Kinematics）
- 逆运动学（Inverse Kinematics）
- DH参数
- 雅可比矩阵

### [避障算法](./避障算法.md)
- 动态窗口法（DWA）
- 人工势场法
- 模型预测控制（MPC）

---

## 学习路径

```
┌─────────────────────────────────────────────────────────────┐
│                    运动规划学习路径                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第1步：理解基本概念                                         │
│  ├── Configuration Space                                    │
│  ├── Workspace vs C-Space                                  │
│  └── 路径 vs 轨迹                                           │
│                                                             │
│  第2步：掌握运动学                                           │
│  ├── 正运动学（DH参数）                                      │
│  ├── 逆运动学（解析/数值解）                                 │
│  └── 雅可比矩阵                                             │
│                                                             │
│  第3步：学习路径规划算法                                     │
│  ├── 图搜索：A*, D*                                         │
│  ├── 采样方法：RRT, PRM                                     │
│  └── 优化方法：CHOMP, TrajOpt                               │
│                                                             │
│  第4步：实践应用                                             │
│  ├── MoveIt! 框架                                           │
│  ├── OMPL 库                                                │
│  └── 实际机器人导航                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 常用工具

| 工具 | 用途 | 链接 |
|------|------|------|
| **OMPL** | 开源运动规划库 | https://ompl.kavrakilab.org/ |
| **MoveIt** | ROS运动规划框架 | https://moveit.ros.org/ |
| **PyBullet** | 物理仿真与规划 | https://pybullet.org/ |

---

## 代码示例：简单的2D路径规划

```python
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

class AStarPlanner:
    """A*路径规划算法实现"""
    
    def __init__(self, obstacle_map, resolution=0.1):
        self.map = obstacle_map
        self.resolution = resolution
        self.height, self.width = obstacle_map.shape
        
    def heuristic(self, a, b):
        """启发式函数：欧几里得距离"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node):
        """获取8连通邻域"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                if self.map[nx, ny] == 0:  # 无障碍
                    neighbors.append((nx, ny))
        return neighbors
    
    def plan(self, start, goal):
        """执行A*规划"""
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heappop(frontier)[1]
            
            if current == goal:
                # 重构路径
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for next_node in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.heuristic(current, next_node)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)
                    heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        return None  # 未找到路径

# 使用示例
if __name__ == "__main__":
    # 创建简单的障碍物地图
    obstacle_map = np.zeros((50, 50))
    obstacle_map[15:35, 20:25] = 1  # 障碍物
    
    planner = AStarPlanner(obstacle_map)
    path = planner.plan((5, 5), (45, 45))
    
    print(f"找到路径，长度: {len(path)}")
```

---

## 延伸阅读

- **书籍**：《Planning Algorithms》by Steven M. LaValle
- **课程**：MIT 6.832 Underactuated Robotics
- **论文**："Sampling-based Algorithms for Optimal Motion Planning" (RRT*)

---

*本章节持续更新中...*
