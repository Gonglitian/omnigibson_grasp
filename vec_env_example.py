import os
from pprint import pprint
from envs.vec_env import VectorEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.macros import gm

from utils.data_utils import process_sim_data_for_vlm

gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False


def main():
    """向量化环境示例主函数"""
    # 指定配置文件路径

    # 创建向量化环境（2个并行环境）
    num_envs = 2
    vec_env = VectorEnvironment(num_envs=num_envs, config="config/scene_config.yaml")

    # 创建控制器实例用于生成随机动作
    controllers = []
    for i in range(num_envs):
        robot = vec_env.envs[i].robots[0]
        controllers.append(KeyboardRobotController(robot=robot))

    # get robot name list of envs
    robot_names = [env.robots[0].name for env in vec_env.envs]

    while True:
        # 使用KeyboardRobotController的get_random_action方法生成随机动作
        actions = []
        for controller in controllers:
            actions.append(controller.get_random_action())

        # 执行动作
        obs, rewards, terminates, truncates, infos = vec_env.step(actions)

        processed_data = process_sim_data_for_vlm(obs, robot_names)

        pprint(f"processed_data: {processed_data}")


if __name__ == "__main__":
    main()
