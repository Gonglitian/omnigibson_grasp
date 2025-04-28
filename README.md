# 🤖 Robot Desktop Grasping Simulation Project

This project is developed based on the OmniGibson framework for simulating Tiago robot desktop object grasping tasks in a virtual environment. The system supports automatically generating randomly placed objects with intelligent layouts considering the table orientation.

## 📑 Table of Contents

- [✨ Key Features](#-key-features)
- [💻 System Requirements](#-system-requirements)
- [📁 Project Structure](#-project-structure)
- [🧩 Core Modules](#-core-modules)
  - [📦 table_grid_generate.py - Object Generation Engine](#-table_grid_generatepy---object-generation-engine)
  - [🏠 base_env.py - Environment Management System](#-base_envpy---environment-management-system)
  - [🎮 base_env_example.py - Main Program Controller](#-base_env_examplepy---main-program-controller)
  - [🔧 debug.py - Debugging Tools](#-debugpy---debugging-tools)
- [⚙️ Configuration System](#️-configuration-system)
  - [🌐 Environment Configuration](#-environment-configuration)
  - [🤖 Robot Configuration](#-robot-configuration)
  - [🏗️ Object Configuration](#️-object-configuration)
- [🚀 How to Run](#-how-to-run)
  - [⌨️ Keyboard Control Guide](#️-keyboard-control-guide)
- [🛠️ Custom Development Guide](#️-custom-development-guide)
  - [🍎 Adding New Object Categories](#-adding-new-object-categories)
  - [📏 Adjusting Object Generation Layout](#-adjusting-object-generation-layout)
  - [🦾 Customizing Robot Initial Pose](#-customizing-robot-initial-pose)
- [💡 Technical Implementation Details](#-technical-implementation-details)

## ✨ Key Features

- 🧠 **Intelligent Object Generation** - Automatically generates grid-based object layouts based on table size and orientation
- 🎲 **Randomized Placement** - Adds random offsets to the grid base to ensure natural object placement
- ⌨️ **Manual Control Mode** - Supports real-time control of the Tiago robot movement and grasping through keyboard
- 🔍 **Complete Debugging Tools** - Provides coordinate axis visualization, robot state viewing, and camera information display

## 💻 System Requirements

- 🛠️ OmniGibson environment
- 🔥 PyTorch
- 🎮 NVIDIA GPU (recommended)

## 📁 Project Structure

```
project_directory/
├── config/                    # Configuration directory
│   └── scene_config.yaml      # Scene and robot configuration
├── utils/                     # Utility functions
│   ├── table_grid_generate.py # Object generation, grid calculation
│   ├── debug.py               # Debugging tools and status display
│   └── ...                    # Other utility modules
├── envs/                      # Environment modules
│   ├── base_env.py            # Custom environment class
│   ├── vec_env.py             # Vectorized environment support
│   └── ...                    # Additional environment modules
├── base_env_example.py        # Main program entry point
├── vec_env_example.py         # Vectorized environment example
└── README.md                  # Project documentation
```

## 🧩 Core Modules

### 📦 table_grid_generate.py - Object Generation Engine

This module implements the core algorithms for intelligent object generation:

- 📏 `get_table_bbox`: Retrieves the bounding box and orientation information of the table
- 🔄 `generate_grid_positions`: Calculates grid positions, supporting rotation based on table orientation and random offsets
- 🍎 `generate_cluttered_objects`: Generates object configurations, supporting custom categories and models
- ➕ `random_orientation`: Generates random quaternion orientations, supporting axis-aligned mode

### 🏠 base_env.py - Environment Management System

Inherits from OmniGibson's Environment class, adding extended functionality:

- 📥 `load_config`: Loads configuration from YAML file
- 🏗️ `add_cluttered_objects`: Adds cluttered objects to the environment
- 📦 `add_dynamic_objects`: Generic method for adding dynamic objects to the environment
- 🧹 `remove_dynamic_objects`: Removes dynamic objects from the environment
- 🔄 `reset`: Resets the environment and re-adds dynamic objects
- 🤖 `set_robot_init_joint_positions`: Sets robot initial joint positions

### 🎮 base_env_example.py - Main Program Controller

Implements the main program logic and control flow:

- 🚀 Environment initialization and robot loading
- 🦾 Robot initial pose setting
- 🔄 Manual control mode and keyboard mapping
- ⏱️ Main loop execution and event handling

### 🔧 debug.py - Debugging Tools

Provides rich debugging capabilities:

- 📊 `draw_coordinate_axes`: Draws 3D coordinate axes at any position
- 📈 `display_robot_state`: Displays robot joint states and position information
- 📷 `display_camera_info`: Shows camera position and orientation
- 🔵 `draw_point`: Draws visualization points at specified positions
- ⌨️ `setup_debug_keys`: Configures keyboard shortcuts for various debugging functions

## ⚙️ Configuration System

The project uses a YAML configuration file (`scene_config.yaml`) to define scene and robot properties:

### 🌐 Environment Configuration
```yaml
env:
  action_frequency: 30
  physics_frequency: 120
  device: null
```

### 🤖 Robot Configuration
```yaml
robots:
  - type: Tiago
    obs_modalities: [rgb]
    default_arm_pose: "horizontal"
```

### 🏗️ Object Configuration
```yaml
random_table_objects:
  categories: [apple, mug, bowl, can, can_of_beans, can_of_soda]
  num_objects: [1, 1, 1, 1, 1, 1]
  random_models: true
  padding: 0.03
  occupancy_rate: 0.8
  grid_size: 0.15
```

## 🚀 How to Run

Ensure OmniGibson and its dependencies are installed, then run:

```bash
python base_env_example.py
```

For vectorized environment example:
```bash
python vec_env_example.py
```

### ⌨️ Keyboard Control Guide

- 📈 **D key**: Display robot state
- 📷 **C key**: Display camera information
- 🔄 **R key**: Reset environment
- 🚪 **ESC key**: Exit program

## 🛠️ Custom Development Guide

### 🍎 Adding New Object Categories

Modify the `random_table_objects` section in `scene_config.yaml`:

```yaml
random_table_objects:
  categories: [apple, mug, bowl, banana, orange, book]
  num_objects: [2, 1, 1, 1, 2, 1]  # Number of objects per category
```

### 📏 Adjusting Object Generation Layout

Modify the following parameters in `scene_config.yaml`:

```yaml
random_table_objects:
  padding: 0.1        # Edge padding
  occupancy_rate: 0.5 # Table occupancy rate
  grid_size: 0.2      # Grid size
```

### 🦾 Customizing Robot Initial Pose

Set in the `robots` section of `scene_config.yaml`:

```yaml
robots:
  - type: Tiago
    default_arm_pose: "horizontal"  # Options: vertical, diagonal15, diagonal30, diagonal45, horizontal
```

## 💡 Technical Implementation Details

- 🔄 Grid generation algorithm uses quaternion conversion to rotation matrices to ensure object layout aligns with table orientation
- 🎲 Object positions include random offsets to ensure natural placement
- ⚡ Uses PyTorch tensors for batch coordinate calculations, improving performance
- 🧪 Utilizes OmniGibson's physics engine to ensure realistic physical interactions