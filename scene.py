import torch as th
import omnigibson as og
import yaml
from omnigibson.macros import gm
from utils import generate_cluttered_objects, add_dynamic_objects, wrap_reset_for_dynamic_objects
from omnigibson.objects import DatasetObject
from omnigibson.systems.micro_particle_system import FluidSystem  # 更新为正确的导入
from typing import List

# 不使用GPU动力学，使用flatcache来提高性能
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_TRANSITION_RULES = False


def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def preprocess_config(cfg):
    # 检查是否需要添加随机物品
    custom_cfgs = {}
    custom_cfgs["random_table_objects"] = cfg.get("random_table_objects", {})
    return cfg, custom_cfgs


def postprocess_config(custom_cfgs, env):
    # stop simulation
    og.sim.pause()
    if custom_cfgs["random_table_objects"]:
        env.dynamic_objects = add_dynamic_objects(custom_cfgs, env)
    og.sim.play()


def create_environment(yaml_path):
    """创建并设置环境

    参数:
        cfg (str or dict): 可以是配置文件路径(str)或直接是配置字典(dict)
                           如果为None，则使用默认配置路径

    返回:
        og.Environment: 创建的环境
    """
    config_dict = load_config(yaml_path)
    config_dict, custom_cfgs = preprocess_config(config_dict)
    print("正在创建环境，这可能需要一些时间...")

    # 创建环境
    env = og.Environment(configs=config_dict)

    # 为环境添加自定义重置方法
    wrap_reset_for_dynamic_objects(env, custom_cfgs)

    # 等待场景稳定
    print("等待场景稳定...")
    robot = env.robots[0]
    for _ in range(30):
        # 创建全零动作
        zero_action = th.zeros(robot.action_dim)
        env.step(zero_action)
    print("场景稳定完成")

    postprocess_config(custom_cfgs, env)
    print("环境创建完成！")
    # 设置相机位置，使其指向机器人
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.450, 1.443, 1.678]),
        orientation=th.tensor([-0.075, 0.519, 0.843, -0.117]),
    )

    return env
