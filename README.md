# 🤖 机器人桌面抓取模拟项目

这个项目基于OmniGibson框架开发，用于在虚拟环境中模拟Tiago机器人执行桌面物体抓取任务。系统支持自动生成随机摆放的物品，并考虑桌子朝向进行智能布局。

## 📑 目录

- [✨ 主要特性](#-主要特性)
- [💻 系统要求](#-系统要求)
- [📁 项目结构](#-项目结构)
- [🧩 核心模块详解](#-核心模块详解)
  - [📦 utils.py - 物品生成引擎](#-utilspy---物品生成引擎)
  - [🏠 scene.py - 场景管理系统](#-scenepy---场景管理系统)
  - [🎮 main.py - 主程序控制器](#-mainpy---主程序控制器)
  - [🔧 debug.py - 调试辅助工具](#-debugpy---调试辅助工具)
- [⚙️ 配置系统](#️-配置系统)
  - [🌐 环境配置](#-环境配置)
  - [🤖 机器人配置](#-机器人配置)
  - [🏗️ 物体配置](#️-物体配置)
- [🚀 运行方法](#-运行方法)
  - [⌨️ 键盘控制说明](#️-键盘控制说明)
- [🛠️ 自定义开发指南](#️-自定义开发指南)
  - [🍎 添加新物品类别](#-添加新物品类别)
  - [📏 调整物品生成布局](#-调整物品生成布局)
  - [🦾 自定义机器人初始姿势](#-自定义机器人初始姿势)
- [💡 技术实现细节](#-技术实现细节)

## ✨ 主要特性

- 🧠 **智能物品生成** - 根据桌面尺寸和朝向自动生成网格化物品布局
- 🎲 **随机化摆放** - 在网格基础上添加随机偏移，确保摆放效果自然
- ⌨️ **手动控制模式** - 支持通过键盘实时控制Tiago机器人移动和抓取
- 🔍 **完整调试工具** - 提供坐标轴可视化、机器人状态查看和相机信息显示等功能

## 💻 系统要求

- 🛠️ OmniGibson环境
- 🔥 PyTorch
- 🎮 NVIDIA GPU (推荐)

## 📁 项目结构

```
grasp_project/
├── config/                    # 配置文件目录
│   └── scene_config.yaml      # 场景和机器人配置
├── utils.py                   # 工具函数(物品生成、网格计算等)
├── scene.py                   # 场景创建和环境设置
├── main.py                    # 主程序入口
├── debug.py                   # 调试工具和状态显示
└── README.md                  # 项目说明文档
```

## 🧩 核心模块详解

### 📦 utils.py - 物品生成引擎

该模块实现了智能物品生成的核心算法：

- 📏 `get_table_bbox`: 获取桌子的包围盒和朝向信息
- 🔄 `generate_grid_positions`: 计算网格位置，支持考虑桌子朝向的旋转和随机偏移
- 🍎 `generate_cluttered_objects`: 生成物品配置，支持自定义类别和模型
- ➕ `add_dynamic_objects`: 向场景添加物体并处理物理模拟

### 🏠 scene.py - 场景管理系统

负责OmniGibson环境的创建和配置：

- 📥 `load_config`: 从YAML文件加载配置
- 🏗️ `create_environment`: 创建并设置环境
- 🔄 `postprocess_config`: 处理自定义配置，如添加随机物品
- 🔄 `wrap_reset_for_dynamic_objects`: 确保场景重置时能正确处理动态物体

### 🎮 main.py - 主程序控制器

实现主要的程序逻辑和控制流程：

- 🚀 环境初始化和机器人加载
- 🦾 机器人初始姿态设置
- 🔄 手动/自动控制模式切换
- ⏱️ 主循环执行和事件处理

### 🔧 debug.py - 调试辅助工具

提供丰富的调试功能：

- 📊 `draw_coordinate_axes`: 在任意位置绘制3D坐标轴
- 📈 `display_robot_state`: 显示机器人关节状态和位置信息
- 📷 `display_camera_info`: 显示相机位置和朝向
- ⌨️ `setup_debug_keys`: 配置快捷键绑定各种调试功能

## ⚙️ 配置系统

项目使用YAML配置文件(`scene_config.yaml`)定义场景和机器人属性：

### 🌐 环境配置
```yaml
env:
  action_frequency: 30
  physics_frequency: 120
  device: null
```

### 🤖 机器人配置
```yaml
robots:
  - type: Tiago
    obs_modalities: [rgb]
    default_arm_pose: "horizontal"
```

### 🏗️ 物体配置
```yaml
random_table_objects:
  categories: [apple, mug, bowl, banana]  # 可选物品类别
  num_objects: 8                          # 物品数量
  random_models: true                     # 随机选择模型
```

## 🚀 运行方法

确保已安装OmniGibson及其依赖项，然后运行：

```bash
python main.py
```

### ⌨️ 键盘控制说明

- 📈 **D键**: 显示机器人状态
- 📷 **C键**: 显示相机信息
- 🔄 **R键**: 重置环境
- 🚪 **ESC键**: 退出程序

## 🛠️ 自定义开发指南

### 🍎 添加新物品类别

修改`scene_config.yaml`中的`random_table_objects`部分：

```yaml
random_table_objects:
  categories: [apple, mug, bowl, banana, orange, book]
  num_objects: [2, 1, 1, 1, 2, 1]  # 每类物品数量
```

### 📏 调整物品生成布局

在`scene_config.yaml`中修改以下参数：

```yaml
random_table_objects:
  padding: 0.1        # 边缘填充
  occupancy_rate: 0.5 # 桌面占用率
  grid_size: 0.2      # 网格大小
```

### 🦾 自定义机器人初始姿势

在`scene_config.yaml`中的`robots`部分设置：

```yaml
robots:
  - type: Tiago
    default_arm_pose: "horizontal"  # 可选值: vertical, diagonal15, diagonal30, diagonal45, horizontal
```

## 💡 技术实现细节

- 🔄 网格生成算法使用四元数转换为旋转矩阵，确保物品布局与桌子朝向一致
- 🎲 物品位置包含随机偏移确保自然摆放效果
- ⚡ 使用PyTorch张量进行批量坐标计算，提高性能
- 🧪 通过OmniGibson的物理引擎确保真实的物理交互