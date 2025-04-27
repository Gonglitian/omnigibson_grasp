import copy
import omnigibson as og
from envs.base_env import BaseEnvironment, load_config, preprocess_config

class VectorEnvironment:
    """
    自定义向量化环境类，用于并行运行多个CustomEnvironment实例
    """
    
    def __init__(self, num_envs, config, custom_cfgs=None):
        """
        初始化向量化环境
        
        参数:
            num_envs (int): 环境实例数量
            config (dict or str): 环境配置或配置文件路径
            custom_cfgs (dict): 自定义配置字典
        """
        self.num_envs = num_envs
        
        # 处理配置
        if isinstance(config, str) and config.endswith(('.yaml', '.yml')):
            config = load_config(config)
            if custom_cfgs is None:
                config, custom_cfgs = preprocess_config(config)
        
        # 如果当前有正在运行的模拟器，停止它
        if og.sim is not None:
            og.sim.stop()
            
        print(f"正在创建 {num_envs} 个并行环境，这可能需要一些时间...")
        
        # 创建多个环境实例
        self.envs = []
        for i in range(num_envs):
            print(f"创建环境 {i+1}/{num_envs}...")
            env = BaseEnvironment(
                configs=copy.deepcopy(config), 
                custom_cfgs=copy.deepcopy(custom_cfgs) if custom_cfgs is not None else None,
                in_vec_env=True,  # 标记为向量化环境中的实例
                stabilize_scene=False,  # 不需要在每个环境中单独稳定场景
                set_initial_camera=False  # 不需要设置相机位置
            )
            self.envs.append(env)
        
        # 启动模拟器并完成所有环境的加载
        og.sim.play()
        for env in self.envs:
            # 调用OmniGibson环境的post_play_load方法
            if hasattr(env, 'post_play_load'):
                env.post_play_load()
        
        print("向量化环境创建完成！")
    
    def step(self, actions):
        """
        执行一步模拟
        
        参数:
            actions (list): 每个环境的动作列表
            
        返回:
            tuple: (observations, rewards, terminates, truncates, infos)
        """
        observations, rewards, terminates, truncates, infos = [], [], [], [], []
        
        # 预处理步骤
        for i, action in enumerate(actions):
            self.envs[i]._pre_step(action)
        
        # 执行模拟步骤
        og.sim.step()
        
        # 后处理步骤
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = self.envs[i]._post_step(action)
            observations.append(obs)
            rewards.append(reward)
            terminates.append(terminated)
            truncates.append(truncated)
            infos.append(info)
            
        return observations, rewards, terminates, truncates, infos
    
    def reset(self, get_obs=True):
        """
        重置所有环境
        
        参数:
            get_obs (bool): 是否获取观察结果
            
        返回:
            list: 如果get_obs为True，返回所有环境的观察结果列表
        """
        results = []
        
        # 重置每个环境
        for env in self.envs:
            result = env.reset(get_obs=get_obs)
            if get_obs:
                results.append(result)
        
        # 如果需要获取观察结果，返回结果列表
        if get_obs:
            return results
    
    def close(self):
        """
        关闭向量化环境
        """
        # 清理资源
        for env in self.envs:
            # 移除动态对象
            if hasattr(env, 'remove_dynamic_objects'):
                env.remove_dynamic_objects()
        
        # 停止模拟器
        if og.sim is not None:
            og.sim.stop()
    
    def __len__(self):
        """
        返回环境数量
        """
        return self.num_envs