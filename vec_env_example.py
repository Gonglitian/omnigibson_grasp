import os
import time

from envs.vec_env import VectorEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController

def main():
    """向量化环境示例主函数"""
    # 指定配置文件路径
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(current_dir, "config", "scene_config.yaml")
    
    # 创建向量化环境（2个并行环境）
    num_envs = 2
    print(f"创建包含 {num_envs} 个并行环境的向量化环境...")
    vec_env = VectorEnvironment(num_envs=num_envs, config=config_path)
    
    # 创建控制器实例用于生成随机动作
    controllers = []
    for i in range(num_envs):
        robot = vec_env.envs[i].robots[0]
        controllers.append(KeyboardRobotController(robot=robot))
    
    # 执行一些随机动作
    num_steps = 10
    print(f"执行 {num_steps} 步随机动作...")
    
    for step in range(num_steps):
        print(f"步骤 {step+1}/{num_steps}")
        
        # 使用KeyboardRobotController的get_random_action方法生成随机动作
        actions = []
        for controller in controllers:
            actions.append(controller.get_random_action())
        
        # 执行动作
        observations, rewards, terminates, truncates, infos = vec_env.step(actions)
        
        # 输出一些信息
        for i in range(num_envs):
            robot_pos = vec_env.envs[i].robots[0].get_position_orientation()[0]
            print(f"  环境 {i}: 机器人位置 [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}], 奖励 {rewards[i]:.2f}")
        
        # 等待一小段时间，以便观察
        time.sleep(0.1)
    
    # 关闭环境和清理资源
    print("关闭向量化环境...")
    vec_env.close()
    print("示例完成！")

if __name__ == "__main__":
    main() 