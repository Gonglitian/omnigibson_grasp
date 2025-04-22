import torch as th
import omnigibson as og
from env import CustomEnvironment
import os
from omnigibson.utils.ui_utils import KeyboardRobotController
# 从debug模块导入调试功能
from debug import setup_debug_keys


def main():
    """主函数"""
    # 直接指定配置文件路径，不使用命令行参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config", "scene_config.yaml")

    # 使用自定义环境类创建环境
    env = CustomEnvironment(config_path)

    # 获取机器人实例（现在机器人的关节位置已经在环境创建过程中设置好了）
    robot = env.robots[0]

    print("正在设置键盘控制器...")

    # 创建键盘控制器（确保键盘事件处理被注册）
    keyboard_controller = KeyboardRobotController(robot=robot)

    # 设置调试按键
    setup_debug_keys(keyboard_controller, robot, env)

    # 显示键盘控制信息
    keyboard_controller.print_keyboard_teleop_info()
        # 进入手动控制循环
    while True:
        # 获取键盘控制动作
        action = keyboard_controller.get_teleop_action()
        # 执行动作
        env.step(action)


if __name__ == "__main__":
    main()
