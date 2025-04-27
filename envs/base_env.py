import torch as th
import omnigibson as og
import yaml
from omnigibson.envs.env_base import Environment
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.macros import gm
from omnigibson.utils.python_utils import create_class_from_registry_and_config
import copy

# 不使用GPU动力学，使用flatcache来提高性能
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = False

def load_config(config_path):
    """从YAML文件加载配置
    
    参数:
        config_path (str): 配置文件路径
        
    返回:
        dict: 加载的配置字典
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def preprocess_config(cfg):
    """预处理配置
    
    参数:
        cfg (dict): 配置字典
        
    返回:
        tuple: (处理后的配置字典, 自定义配置字典)
    """
    # 检查是否需要添加随机物品
    custom_cfgs = {}
    custom_cfgs["random_table_objects"] = cfg.get("random_table_objects", {})
    return cfg, custom_cfgs

class BaseEnvironment(Environment):
    

    """
    自定义环境类，继承自OmniGibson的Environment类，
    主要添加了动态对象管理功能。
    """

    def __init__(self, configs, custom_cfgs=None, in_vec_env=False, stabilize_scene=True, set_initial_camera=True):
        """
        初始化自定义环境
        
        参数:
            configs (str or dict or list of str or dict): 配置文件路径或配置字典
            custom_cfgs (dict): 自定义配置字典，用于动态对象生成
            in_vec_env (bool): 是否在向量化环境中使用
            stabilize_scene (bool): 是否在初始化后执行场景稳定步骤
            set_initial_camera (bool): 是否设置初始相机位置
        """
        
        # 如果configs是文件路径，加载配置文件
        if isinstance(configs, str) and configs.endswith(('.yaml', '.yml')):
            configs = load_config(configs)
            # 预处理配置
            if custom_cfgs is None:
                configs, custom_cfgs = preprocess_config(configs)
                
        # 保存自定义配置
        self.custom_cfgs = custom_cfgs if custom_cfgs is not None else {}
        
        # 初始化动态对象列表
        self.dynamic_objects = []
        
        # 打印初始化消息
        print("正在创建环境，这可能需要一些时间...")
        
        # 调用父类初始化方法
        super().__init__(configs=configs, in_vec_env=in_vec_env)
        
        
        # 稳定场景（执行多次零动作）
        if stabilize_scene and self.robots:
            self._stabilize_scene()
            
        # 设置机器人初始关节位置
        if self.robots:
            self.set_robot_init_joint_positions()
            
        # 设置相机位置
        if set_initial_camera:
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor([0.450, 1.443, 1.678]),
                orientation=th.tensor([-0.075, 0.519, 0.843, -0.117]),
            )
            
        print("环境创建完成！")

    def _stabilize_scene(self, steps=30):
        """
        执行多步零动作，使场景稳定下来
        
        参数:
            steps (int): 执行的步数
        """
        print("等待场景稳定...")
        robot = self.robots[0]
        for _ in range(steps):
            # 创建全零动作
            zero_action = th.zeros(robot.action_dim)
            self.step(zero_action)
        print("场景稳定完成")

    def add_cluttered_objects(self):
        """
        向环境中添加随机杂乱的物体
        
        返回:
            list: 添加的物体列表
        """
        if not self.custom_cfgs.get("random_table_objects"):
            return []
            
        # 生成杂乱物体的配置
        # 导入generate_cluttered_objects函数
        from utils import generate_cluttered_objects
        
        objects_cfg = generate_cluttered_objects(
            **self.custom_cfgs["random_table_objects"], 
            env=self, 
            return_cfg=False
        )
        
        # 使用通用方法添加物体
        return self.add_dynamic_objects(objects_cfg)

    def add_dynamic_objects(self, objects_cfg):
        """
        通用方法：向环境中添加动态物体并返回物体列表
        
        参数:
            objects_cfg (list): 物体配置列表，每个配置是一个字典，包含创建物体所需的参数
            
        返回:
            list: 添加的动态物体列表
        """
        new_objects = []
        
        # 暂停模拟以安全地添加物体
        is_playing = og.sim.is_playing()
        if is_playing:
            og.sim.pause()

        try:
            objects_instances = []

            # 创建物体实例
            for obj_cfg in objects_cfg:
                # 确保配置中有type字段
                if "type" not in obj_cfg:
                    print(f"警告: 配置中缺少type字段，无法创建物体")
                    continue
                
                # 复制配置，避免修改原始配置
                cfg_copy = obj_cfg.copy()
                
                # 暂存position和orientation
                position = cfg_copy.pop("position", None)
                orientation = cfg_copy.pop("orientation", None)
                
                # 使用create_class_from_registry_and_config创建对象
                # 这样可以支持不同类型的对象，包括DatasetObject和PrimitiveObject
                try:
                    obj = create_class_from_registry_and_config(
                        cls_name=cfg_copy["type"],
                        cls_registry=REGISTERED_OBJECTS,
                        cfg=cfg_copy,
                        cls_type_descriptor="object",
                    )
                    objects_instances.append((obj, position, orientation))
                except Exception as e:
                    print(f"创建物体时出错: {e}")
                    continue

            # 批量添加所有对象
            for obj, position, orientation in objects_instances:
                try:
                    self.scene.add_object(obj)
                    # 设置位置
                    if position is not None or orientation is not None:
                        obj.set_position_orientation(
                            position=position,
                            orientation=orientation,
                            frame="scene"
                        )
                    new_objects.append(obj)
                except Exception as e:
                    print(f"添加物体时出错: {e}")
        finally:
            # 恢复模拟状态
            if is_playing:
                og.sim.play()
                # 执行一次物理步骤让对象稳定
                og.sim.step_physics()

        # 将新添加的物体加入到动态物体列表中
        self.dynamic_objects.extend(new_objects)
        print(f"成功添加了 {len(new_objects)} 个动态物体")
        
        return new_objects

    def remove_dynamic_objects(self):
        """
        从环境中移除所有动态添加的物体
        """
        objects_to_remove = []
        
        # 收集所有要移除的对象
        for obj in self.dynamic_objects:
            if (obj is not None and 
                hasattr(obj, "scene") and 
                obj.scene is not None):
                objects_to_remove.append(obj)

        # 清空动态对象列表
        self.dynamic_objects = []

        # 如果有对象需要移除，使用batch_remove_objects
        if objects_to_remove:
            try:
                # 暂停模拟以安全地移除物体
                is_playing = og.sim.is_playing()
                if is_playing:
                    og.sim.pause()

                # 批量移除所有对象
                og.sim.batch_remove_objects(objects_to_remove)

                # 恢复模拟状态
                if is_playing:
                    og.sim.play()
            except Exception as e:
                print(f"批量移除物体时出错: {e}")

    def reset(self, get_obs=True, **kwargs):
        """
        重置环境，包括移除并重新添加动态物体

        参数:
            get_obs (bool): 是否获取观察结果
            **kwargs: 传递给父类reset方法的其他参数

        返回:
            与父类reset方法相同的返回值
        """
        # 确保模拟器在重置前处于播放状态
        was_paused = False
        if not og.sim.is_playing():
            print("重置前启动模拟状态...")
            was_paused = True
            og.sim.play()

        try:
            # 移除所有动态添加的物体
            self.remove_dynamic_objects()
            
            # 调用父类的reset方法
            result = super().reset(get_obs=False, **kwargs)

            # 重新添加动态物体
            if "random_table_objects" in self.custom_cfgs and self.custom_cfgs["random_table_objects"]:
                self.add_cluttered_objects()

            # 获取观察结果并返回
            if get_obs:
                og.sim.step()
                obs, info = self.get_obs()
                return obs, info
            return result
        finally:
            # 如果之前是暂停状态，恢复到暂停状态
            if was_paused and og.sim.is_playing():
                og.sim.pause()

    def set_robot_init_joint_positions(self, joint_positions=None):
        """
        设置机器人的初始关节位置
        
        参数:
            joint_positions (dict or tensor, optional): 关节位置映射或张量。
                                                       如果为None，将设置默认的头部转向位置。
        """
        if not self.robots:
            print("环境中没有机器人，无法设置关节位置")
            return
            
        robot = self.robots[0]
        
        if joint_positions is None:
            # 默认设置头部转向关节
            joint_positions = robot.get_joint_positions()
            head_pan_idx = list(robot.joints.keys()).index("head_2_joint")
            joint_positions[head_pan_idx] = -1
            
        # 设置关节位置
        robot.set_joint_positions(joint_positions)
        print("已设置机器人初始关节位置")
        
        # 保存当前关节位置为默认值，这样在后续reset时会使用这些位置值
        robot.reset_joint_pos = joint_positions.clone()
        print("已将当前关节位置设为默认重置位置")
        
        return joint_positions

