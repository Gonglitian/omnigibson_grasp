from pprint import pprint
from envs.vec_env import VecEnv
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.macros import gm

gm.RENDER_VIEWER_CAMERA = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = False


def main():
    """向量化环境示例主函数"""
    # 指定配置文件路径

    # 创建向量化环境（2个并行环境）
    num_envs = 2
    vec_env = VecEnv(num_envs=num_envs, config="config/scene_config.yaml")

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
if __name__ == "__main__":
    main()
