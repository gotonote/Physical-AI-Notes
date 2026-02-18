# ROS/ROS2

ROS (Robot Operating System)是机器人软件开发的重要框架，ROS2是其新一代版本。

## 目录

- [1. ROS概述](#1-ros概述)
- [2. 核心概念](#2-核心概念)
- [3. ROS2新特性](#3-ros2新特性)
- [4. 机器人控制](#4-机器人控制)
- [5. 实践](#5-实践)

---

## 1. ROS概述

### 1.1 架构

```
┌─────────────────────────────────────────┐
│           ROS 架构                      │
├─────────────────────────────────────────┤
│  应用层                                 │
│  ├── 导航 (Navigation)                  │
│  ├── 感知 (Perception)                  │
│  └── 控制 (Control)                     │
├─────────────────────────────────────────┤
│  通信层                                 │
│  ├── Topic (发布/订阅)                   │
│  ├── Service (请求/响应)                 │
│  └── Action (异步目标)                   │
├─────────────────────────────────────────┤
│  硬件抽象层                             │
│  └── 驱动 (Drivers)                      │
└─────────────────────────────────────────┘
```

### 1.2 安装

```bash
# ROS2 Humble (Ubuntu 22.04)
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

---

## 2. 核心概念

### 2.1 节点(Node)

```python
# Python节点示例
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # 创建发布者
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 创建订阅者
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
    def odom_callback(self, msg):
        self.get_logger().info(f"Position: {msg.pose.pose.position}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### 2.2 消息(Topic)

```bash
# 查看话题
ros2 topic list

# 监听话题
ros2 topic echo /odom

# 发布消息
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}"
```

---

## 3. ROS2新特性

### 3.1 DDS中间件

```yaml
# config/rmw_cyclonedds.yaml
rmw_cyclonedds:
  domain_id: 0
  medium: UDP
```

### 3.2 Action

```python
# Action服务器
class NavigateActionServer(Node):
    def __init__(self):
        super().__init__('navigate_action')
        
        self._action_server = ActionServer(
            self,
            Navigate,
            'navigate',
            self.execute_callback
        )
        
    def execute_callback(self, goal_handle):
        # 执行导航
        result = Navigate.Result()
        goal_handle.succeed(result)
        return result
```

---

## 4. 机器人控制

### 4.1 运动控制

```python
# Joint控制
from trajectory_msgs.msg import JointTrajectory

def send_joint_trajectory(pub, positions):
    msg = JointTrajectory()
    msg.joint_names = ['joint1', 'joint2', 'joint3']
    point = JointTrajectoryPoint()
    point.positions = positions
    point.time_from_start = Duration(sec=1)
    msg.points = [point]
    pub.publish(msg)
```

---

## 5. 实践

### 5.1 启动文件

```python
# launch/robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_control',
            executable='controller',
            name='robot_controller'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        )
    ])
```

---

*本章节持续更新中...*
