import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Tuple


def process_sim_data_for_vlm(
    vec_obs: List[Dict[str, Any]], envs: List, prompt_template: Optional[str] = None
) -> Dict[str, List]:
    """
    Extract and process observation data from vectorized environment data for multiple environments and robots,
    converting it into a batch format usable by VLM.

    Args:
        vec_obs (List[Dict[str, Any]]): List of observations from vectorized environments, each element corresponds to an environment's observation
        envs (list): List of environment objects
        prompt_template (str, optional): Optional prompt template used to build prompt text

    Returns:
        Dict: Dictionary containing the following key-value pairs:
            - rgb_images (List): List of PIL images
            - proprio_vectors (List): List of proprioception data
            - prompts (List): List of prompt texts generated for each robot
            - messages (List): List of messages prepared for VLM
    """
    rgb_images = []
    proprio_vectors = []
    prompts = []
    messages = []
    robot_names = [env.robots[0].name for env in envs]
    # Iterate through all environments
    for env_idx, obs in enumerate(vec_obs):
        # Iterate through all robots
        for robot_name in robot_names:
            if robot_name not in obs:
                continue

            # Extract RGB image
            rgb_tensor = obs[robot_name][f"{robot_name}:eyes:Camera:0"]["rgb"]
            # Extract proprioception data
            proprio_tensor = obs[robot_name]["proprio"]

            # Process RGB image
            if rgb_tensor.shape[2] == 4:  # If there's an alpha channel, remove it
                rgb_tensor = rgb_tensor[:, :, :3]
            # Convert to PIL image
            rgb_img = Image.fromarray(rgb_tensor.numpy())

            # Convert proprioception tensor to numpy
            proprio_vector = proprio_tensor.numpy()

            # Build prompt text
            default_prompt = f"Based on the robot's proprioception data: {proprio_vector}, determine the next action."
            prompt = default_prompt
            if prompt_template:
                prompt = prompt_template.format(proprio=proprio_vector)

            # Create VLM input message
            message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }

            # Add to corresponding lists
            rgb_images.append([rgb_img])
            messages.append([message])
            proprio_vectors.append(proprio_vector)
            prompts.append(prompt)

    # Organize data into batches
    batch_data = {
        "rgb_images": rgb_images,
        "messages": messages,
        "proprio_vectors": proprio_vectors,
        "prompts": prompts,
    }

    return batch_data


def batch_inference_with_vlm(
    images_list: List[List[Image.Image]],
    messages_list: List[List[Dict[str, Any]]],
    model: Any,
    tokenizer: Any,
) -> List[str]:
    """
    Perform batch inference with a Vision Language Model (VLM).

    Args:
        images_list (List[List[Image.Image]]): List of image lists for each sample
        messages_list (List[List[Dict[str, Any]]]): List of message lists for each sample
        model (Any): The VLM model
        tokenizer (Any): The tokenizer for the model

    Returns:
        List[str]: List of inference results
    """

    inference_results = []
    tokenizer.padding_side = "left"
    input_text = tokenizer.apply_chat_template(
        messages_list, add_generation_prompt=True
    )
    inputs = tokenizer(
        images_list,
        input_text,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")

    # Inference
    results = model.generate(
        **inputs, max_new_tokens=128, use_cache=True, temperature=1.3, min_p=0.1
    )
    outputs = tokenizer.batch_decode(results, skip_special_tokens=True)
    for o in outputs:
        if "assistant\n" in o:
            inference_results.append(o.split("assistant\n")[-1].strip())
    return inference_results
