import omnigibson as og
from utils import generate_grid_positions, get_table_bbox
from debug import visualize_grid_positions, draw_coordinate_axes


def main():
    """
    演示如何使用debug.py中的函数来可视化generate_grid_positions生成的点
    """
    print("启动OmniGibson...")

    # 配置环境
    env_config = {
        "scene": {
            "type": "Scene",
            "scene_model": "Rs_int",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "fixed_base": True,
                "position": [0, 0, 1],
            }
        ],
    }

    # 启动仿真并加载环境
    env = og.Environment(configs=env_config)
    og.sim.enable_viewer_camera_teleoperation()
    print("环境已加载")

    # 获取桌子的包围盒信息
    bbox_center, bbox_extent, table_height, table_orientation = get_table_bbox(
        env, "table"
    )
    if bbox_center is not None:
        print(f"获取到桌子的包围盒信息: ")
        print(f"中心: {bbox_center}")
        print(f"尺寸: {bbox_extent}")
        print(f"高度: {table_height}")
        print(f"方向: {table_orientation}")

        # 绘制坐标轴以显示参考系统
        # draw_coordinate_axes(origin=bbox_center.tolist(), scale=0.3)

        # 生成网格位置
        grid_size = 0.15  # 网格大小，单位为米
        padding = 0  # 边缘填充，单位为米
        occupancy_rate = 0.8  # 网格占用率

        positions, available_positions, grid_positions = generate_grid_positions(
            bbox_center,
            bbox_extent,
            table_orientation,
            grid_size=grid_size,
            padding=padding,
            occupancy_rate=occupancy_rate,
        )

        print(f"生成的网格点数量: {available_positions}")

        # 可视化网格点
        spheres_1 = visualize_grid_positions(
            env,
            positions,
            color=(0.0, 0.5, 1.0, 0.7),  # 蓝色，半透明
            radius=0.01,  # 小球半径
            prefix="obj_point",
        )

        spheres_2 = visualize_grid_positions(
            env,
            grid_positions,
            color=(1, 0, 0, 0.7),  # 红色
            radius=0.01,  # 小球半径
            prefix="grid_point",
        )

        print("按Esc键退出...")
        # 渲染循环
        while True:
            og.sim.step()

    # 关闭OmniGibson
    env.close()
    og.shutdown()


if __name__ == "__main__":
    main()
