import torch
import numpy as np
from PIL import Image

def process_sim_data_for_vlm(vec_obs, robot_names, prompt_template=None):
    """
    从向量化环境数据中提取并处理多个环境和机器人的观测数据，转换为VLM可用的批处理格式
    
    参数:
        vec_obs (list): 向量化环境的观测数据列表，每个元素对应一个环境的观测
        robot_names (list): 机器人名称列表
        prompt_template (str, optional): 可选的提示模板，用于构建提示文本
        
    返回:
        dict: 包含以下键值对的字典:
            - rgb_images (list): PIL图像列表
            - proprio_vectors (list): 本体感知数据列表
            - prompts (list): 为每个机器人生成的提示文本列表
            - messages (list): 为VLM准备的消息列表
            - env_indices (list): 对应的环境索引列表
    """
    rgb_images = []
    proprio_vectors = []
    prompts = []
    messages = []
    env_indices = []
    
    # 遍历所有环境
    for env_idx, obs in enumerate(vec_obs):
        # 遍历所有机器人
        for robot_name in robot_names:
            if robot_name not in obs:
                continue
                
            # 提取RGB图像
            rgb_tensor = obs[robot_name][f"{robot_name}:eyes:Camera:0"]["rgb"]
            # 提取本体感知数据
            proprio_tensor = obs[robot_name]["proprio"]
            
            # 处理RGB图像
            if rgb_tensor.shape[2] == 4:  # 如果有alpha通道，去除它
                rgb_tensor = rgb_tensor[:, :, :3]
            # 转换为PIL图像
            rgb_img = Image.fromarray(rgb_tensor.numpy())
            
            # 本体感知张量转换为numpy
            proprio_vector = proprio_tensor.numpy()
            
            # 构建提示文本
            default_prompt = f"基于机器人的本体感知数据:{proprio_vector}，确定下一步行动。"
            prompt = default_prompt
            if prompt_template:
                prompt = prompt_template.format(proprio=proprio_vector)
            
            # 创建VLM输入消息
            message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_img},
                    {"type": "text", "text": prompt}
                ],
            }
            
            # 添加到对应列表
            rgb_images.append(rgb_img)
            proprio_vectors.append(proprio_vector)
            prompts.append(prompt)
            messages.append(message)
            env_indices.append(env_idx)
    
    # 整合数据为批次
    batch_data = {
        "rgb_images": rgb_images,
        "proprio_vectors": proprio_vectors,
        "prompts": prompts,
        "messages": messages,
        "env_indices": env_indices
    }
    
    return batch_data

def batch_inference_with_vlm(batch_data, model, processor, process_vision_info, max_new_tokens=128):
    """
    使用视觉语言模型对批处理数据进行推理
    
    参数:
        batch_data (dict): 由process_sim_data_for_vlm函数生成的批处理数据
        model: VLM模型
        processor: VLM处理器
        process_vision_info (function): 处理视觉信息的函数
        max_new_tokens (int): 生成的最大新token数
        
    返回:
        list: 模型生成的文本结果列表
    """
    results = []
    
    # 处理每个机器人的数据
    for i, message in enumerate(batch_data["messages"]):
        messages = [message]
        
        # 应用聊天模板
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 准备模型输入
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # 将输入移至适当的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 执行推理
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # 解码生成的输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        results.append(output_text[0])
    
    return results 