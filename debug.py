import torch as th
import numpy as np
import omnigibson as og
from omnigibson.utils.ui_utils import KeyboardRobotController, draw_line, clear_debug_drawing


def draw_coordinate_axes(origin=[0, 0, 0], scale=1.0):
    """
    在指定位置绘制XYZ坐标轴

    Args:
        origin: 坐标轴原点位置
        scale: 坐标轴的长度缩放因子
    """
    # 确保origin是numpy数组
    origin = np.array(origin)

    # 坐标轴长度
    axis_length = scale

    # 绘制X轴 (红色)
    x_end = origin + np.array([axis_length, 0, 0])
    draw_line(origin, x_end, color=(1.0, 0.0, 0.0, 1.0), size=2.0)

    # 绘制Y轴 (绿色)
    y_end = origin + np.array([0, axis_length, 0])
    draw_line(origin, y_end, color=(0.0, 1.0, 0.0, 1.0), size=2.0)

    # 绘制Z轴 (蓝色)
    z_end = origin + np.array([0, 0, axis_length])
    draw_line(origin, z_end, color=(0.0, 0.0, 1.0, 1.0), size=2.0)

    print(f"坐标轴已绘制于位置 {origin}，比例 {scale}")
    print("X轴 - 红色，Y轴 - 绿色，Z轴 - 蓝色")


# 注册自定义按键绑定，用于重新绘制坐标轴
def redraw_axes(env):
    clear_debug_drawing()
    draw_coordinate_axes(origin=[0, 0, 0], scale=1.0)
    table_pos = env.scene.object_registry("name", "table").get_position()
    draw_coordinate_axes(origin=table_pos.tolist(), scale=0.5)
    apple_pos = env.scene.object_registry("name", "apple").get_position()
    draw_coordinate_axes(origin=apple_pos.tolist(), scale=0.2)
    print("已重新绘制坐标轴")


def display_robot_state(robot):
    """显示机器人状态信息"""
    # 获取机器人的本体感知信息
    proprio_dict = robot._get_proprioception_dict()

    # 获取位置和方向
    position = proprio_dict["robot_pos"].numpy()

    # 获取关节位置和速度
    joint_positions = proprio_dict["joint_qpos"].numpy()
    joint_velocities = proprio_dict["joint_qvel"].numpy()

    # 获取线速度和角速度
    linear_vel = proprio_dict["robot_lin_vel"].numpy()
    angular_vel = proprio_dict["robot_ang_vel"].numpy()

    # 获取所有关节名称
    joint_names = list(robot.joints.keys())

    # 输出信息
    print("\n===== 机器人状态信息 =====")
    print(f"位置: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    print(f"线速度: [{linear_vel[0]:.3f}, {linear_vel[1]:.3f}, {linear_vel[2]:.3f}]")
    print(f"角速度: [{angular_vel[0]:.3f}, {angular_vel[1]:.3f}, {angular_vel[2]:.3f}]")

    # 显示所有关节的位置和速度
    print("\n关节位置:")
    for i, name in enumerate(joint_names):
        if i < len(joint_positions):
            print(f"  {name}: {joint_positions[i]:.3f}")

    # print("\n关节速度:")
    # for i, name in enumerate(joint_names):
    #     if i < len(joint_velocities):
    #         print(f"  {name}: {joint_velocities[i]:.3f}")

    # 显示各个控制器组的索引
    # print("\n控制器组索引:")
    # try:
    #     if hasattr(robot, 'base_control_idx'):
    #         print(f"  基座控制索引: {robot.base_control_idx.tolist()}")
    #     if hasattr(robot, 'arm_control_idx'):
    #         for arm_name, idx in robot.arm_control_idx.items():
    #             print(f"  手臂 {arm_name} 控制索引: {idx.tolist()}")
    #     if hasattr(robot, 'gripper_control_idx'):
    #         for gripper_name, idx in robot.gripper_control_idx.items():
    #             print(f"  抓手 {gripper_name} 控制索引: {idx.tolist()}")
    #     if hasattr(robot, 'trunk_control_idx'):
    #         print(f"  躯干控制索引: {robot.trunk_control_idx.tolist()}")
    #     if hasattr(robot, 'camera_control_idx'):
    #         print(f"  相机控制索引: {robot.camera_control_idx.tolist()}")
    # except Exception as e:
    #     print(f"  获取控制索引时出错: {e}")

    print("==========================\n")


def display_camera_info():
    """显示当前相机位置和朝向信息"""
    # 获取相机位置和朝向
    camera_position = og.sim.viewer_camera.get_position().numpy()
    camera_orientation = og.sim.viewer_camera.get_orientation().numpy()

    # 计算相机的前向向量（用于显示相机朝向）
    # 四元数转换为前向向量的近似计算
    x = camera_orientation[0]
    y = camera_orientation[1]
    z = camera_orientation[2]
    w = camera_orientation[3]

    # 计算前向向量 (z轴方向)
    forward_x = 2 * (x * z + w * y)
    forward_y = 2 * (y * z - w * x)
    forward_z = 1 - 2 * (x * x + y * y)

    # 输出信息
    print("\n===== 相机信息 =====")
    print(f"位置: [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]")
    print(
        f"四元数朝向: [{camera_orientation[0]:.3f}, {camera_orientation[1]:.3f}, {camera_orientation[2]:.3f}, {camera_orientation[3]:.3f}]"
    )
    print(f"前向向量: [{forward_x:.3f}, {forward_y:.3f}, {forward_z:.3f}]")
    print("代码调用格式:")
    print(f"og.sim.viewer_camera.set_position_orientation(")
    print(f"    position=th.tensor([{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]),")
    print(
        f"    orientation=th.tensor([{camera_orientation[0]:.3f}, {camera_orientation[1]:.3f}, {camera_orientation[2]:.3f}, {camera_orientation[3]:.3f}]),"
    )
    print(f")")
    print("=====================\n")


def setup_debug_keys(action_generator, robot, env):
    """设置调试按键绑定"""
    # 注册自定义按键绑定，用于显示机器人状态
    action_generator.register_custom_keymapping(
        key=og.lazy.carb.input.KeyboardInput.D,
        description="显示机器人状态信息",
        callback_fn=lambda: display_robot_state(robot),
    )

    # 注册自定义按键绑定，用于重置环境
    action_generator.register_custom_keymapping(
        key=og.lazy.carb.input.KeyboardInput.R,
        description="重置环境",
        callback_fn=lambda: env.reset(),
    )

    # 注册自定义按键绑定，用于显示相机信息
    action_generator.register_custom_keymapping(
        key=og.lazy.carb.input.KeyboardInput.C,
        description="显示当前相机位置和朝向",
        callback_fn=display_camera_info,
    )

    # 注册自定义按键绑定，用于清除并重新绘制坐标轴
    action_generator.register_custom_keymapping(
        key=og.lazy.carb.input.KeyboardInput.X,
        description="清除并重新绘制坐标轴",
        callback_fn=lambda: redraw_axes(env),
    )
