import torch as th
import omnigibson as og
from scene import create_environment
import os
from omnigibson.utils.ui_utils import KeyboardRobotController

# 从debug模块导入调试功能
from debug import draw_coordinate_axes, display_robot_state
from debug import setup_debug_keys


def main():
    """主函数"""
    # 直接指定配置文件路径，不使用命令行参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config", "scene_config.yaml")

    env = create_environment(config_path)

    # 获取机器人实例
    robot = env.robots[0]

    # 设置特定关节的初始位置
    joint_positions = robot.get_joint_positions()
    head_pan_idx = list(robot.joints.keys()).index("head_2_joint")
    joint_positions[head_pan_idx] = -1
    robot.set_joint_positions(joint_positions)
    print("已设置机器人初始关节位置")

    # 保存当前关节位置为默认值，这样在后续reset时会使用这些位置值
    robot.reset_joint_pos = joint_positions.clone()
    print("已将当前关节位置设为默认重置位置")

    # 绘制坐标轴辅助调试
    # draw_coordinate_axes(origin=[0, 0, 0], scale=1.0)

    # 显示初始机器人状态
    # display_robot_state(robot)

    # 默认运行模式设置
    manual_mode = True  # 默认为手动模式

    # 运行模拟
    if manual_mode:
        print("启用手动控制模式")
        print("正在设置键盘控制器...")

        # 创建键盘控制器（确保键盘事件处理被注册）
        keyboard_controller = KeyboardRobotController(robot=robot)

        # 设置调试按键
        setup_debug_keys(keyboard_controller, robot, env)

        # 显示键盘控制信息
        # keyboard_controller.print_keyboard_teleop_info()

        try:
            # 进入手动控制循环
            while True:
                # 获取键盘控制动作
                action = keyboard_controller.get_teleop_action()
                # 执行动作
                env.step(action)
        except KeyboardInterrupt:
            print("用户手动终止模拟")
    else:
        print("启用自动抓取模式")
        # 尝试自动抓取苹果
        while True:
            # TODO: 实现自动抓取逻辑
            action = th.zeros(robot.action_dim)
            obs, reward, done, info = env.step(action)

    print("模拟完成")
    og.clear()


if __name__ == "__main__":
    main()
