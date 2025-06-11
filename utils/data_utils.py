"""
Data processing utilities for VLM inference with OmniGibson environments.

This module provides efficient functions to process simulation data and perform
batch inference with Vision Language Models.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VLMBatch:
    """Data structure for VLM batch processing."""
    images: List[Image.Image]
    messages: List[Dict[str, Any]]
    proprio_data: List[np.ndarray]
    prompts: List[str]
    
    def __len__(self) -> int:
        return len(self.images)


class SimDataProcessor:
    """Efficient processor for simulation data to VLM format conversion."""
    
    def __init__(self, 
                 camera_key_pattern: str = "{robot_name}:eyes:Camera:0",
                 default_prompt_template: str = "Robot observation - proprioception: {proprio}. What should the robot do next?"):
        """
        Initialize the processor.
        
        Args:
            camera_key_pattern: Pattern for camera observation keys
            default_prompt_template: Default template for generating prompts
        """
        self.camera_key_pattern = camera_key_pattern
        self.default_prompt_template = default_prompt_template
    
    def extract_robot_data(self, obs: Dict[str, Any], robot_name: str) -> Optional[Tuple[Image.Image, np.ndarray]]:
        """
        Extract RGB image and proprioception data for a single robot.
        
        Args:
            obs: Observation dictionary
            robot_name: Name of the robot
            
        Returns:
            Tuple of (RGB image, proprioception array) or None if extraction fails
        """
        try:
            robot_obs = obs.get(robot_name)
            if not robot_obs:
                return None
            
            # Extract RGB image
            camera_key = self.camera_key_pattern.format(robot_name=robot_name)
            rgb_data = robot_obs.get(camera_key, {}).get("rgb")
            if rgb_data is None:
                logger.warning(f"No RGB data found for {robot_name}")
                return None
            
            # Process RGB image (remove alpha channel if present)
            if rgb_data.shape[-1] == 4:
                rgb_data = rgb_data[..., :3]
            
            rgb_image = Image.fromarray(rgb_data.numpy() if hasattr(rgb_data, 'numpy') else rgb_data)
            
            # Extract proprioception data
            proprio_data = robot_obs.get("proprio")
            if proprio_data is None:
                logger.warning(f"No proprioception data found for {robot_name}")
                return None
                
            proprio_array = proprio_data.numpy() if hasattr(proprio_data, 'numpy') else np.array(proprio_data)
            
            return rgb_image, proprio_array
            
        except Exception as e:
            logger.error(f"Failed to extract data for {robot_name}: {e}")
            return None
    
    def create_vlm_message(self, prompt: str) -> Dict[str, Any]:
        """Create a VLM input message."""
        return {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    
    def generate_prompt(self, proprio_data: np.ndarray, template: Optional[str] = None) -> str:
        """Generate prompt text from proprioception data."""
        prompt_template = template or self.default_prompt_template
        return prompt_template.format(proprio=proprio_data)
    
    def process_batch(self, 
                     vec_obs: List[Dict[str, Any]], 
                     envs: List,
                     prompt_template: Optional[str] = None) -> VLMBatch:
        """
        Process vectorized observations into VLM batch format.
        
        Args:
            vec_obs: List of observations from vectorized environments
            envs: List of environment objects
            prompt_template: Optional custom prompt template
            
        Returns:
            VLMBatch containing processed data
        """
        # Pre-allocate lists for better performance
        batch_size = len(vec_obs) * len(envs)
        images, messages, proprio_arrays, prompts = [], [], [], []
        
        # Extract robot names once
        robot_names = [env.robots[0].name for env in envs if env.robots]
        
        # Process each observation
        for obs in vec_obs:
            for robot_name in robot_names:
                result = self.extract_robot_data(obs, robot_name)
                if result is None:
                    continue
                    
                rgb_image, proprio_data = result
                
                # Generate prompt and message
                prompt = self.generate_prompt(proprio_data, prompt_template)
                message = self.create_vlm_message(prompt)
                
                # Append to batch
                images.append(rgb_image)
                messages.append(message)
                proprio_arrays.append(proprio_data)
                prompts.append(prompt)
        
        return VLMBatch(
            images=images,
            messages=messages,
            proprio_data=proprio_arrays,
            prompts=prompts
        )


class VLMInference:
    """Efficient VLM inference handler."""
    
    def __init__(self, model, tokenizer, default_gen_config: Optional[Dict] = None):
        """
        Initialize VLM inference handler.
        
        Args:
            model: VLM model
            tokenizer: Model tokenizer
            default_gen_config: Default generation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.default_gen_config = default_gen_config or {
            "max_new_tokens": 128,
            "use_cache": True,
            "temperature": 1.3,
            "min_p": 0.1
        }
        
        # Configure tokenizer
        self.tokenizer.padding_side = "left"
    
    def batch_inference(self, 
                       batch: VLMBatch, 
                       generation_config: Optional[Dict] = None) -> List[str]:
        """
        Perform batch inference on VLM batch data.
        
        Args:
            batch: VLMBatch containing images and messages
            generation_config: Optional generation parameters
            
        Returns:
            List of inference results
        """
        if len(batch) == 0:
            return []
        
        gen_config = generation_config or self.default_gen_config
        
        try:
            # Prepare inputs
            input_text = self.tokenizer.apply_chat_template(
                batch.messages, add_generation_prompt=True
            )
            
            # Batch images properly
            images_batch = [[img] for img in batch.images]
            
            inputs = self.tokenizer(
                images_batch,
                input_text,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to("cuda")
            
            # Generate responses
            with torch.no_grad():
                results = self.model.generate(**inputs, **gen_config)
            
            # Decode and extract responses
            outputs = self.tokenizer.batch_decode(results, skip_special_tokens=True)
            return self._extract_responses(outputs)
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return ["Error"] * len(batch)
    
    def _extract_responses(self, outputs: List[str]) -> List[str]:
        """Extract clean responses from model outputs."""
        responses = []
        for output in outputs:
            if "assistant\n" in output:
                response = output.split("assistant\n")[-1].strip()
            else:
                # Fallback: try to find response after last user prompt
                lines = output.split('\n')
                response = lines[-1].strip() if lines else output.strip()
            responses.append(response)
        return responses


# Convenience functions for backward compatibility
def process_sim_data_for_vlm(
    vec_obs: List[Dict[str, Any]], 
    envs: List, 
    prompt_template: Optional[str] = None
) -> Dict[str, List]:
    """
    Legacy function for backward compatibility.
    
    Returns data in the original format for existing code.
    """
    processor = SimDataProcessor()
    batch = processor.process_batch(vec_obs, envs, prompt_template)
    
    return {
        "rgb_images": [[img] for img in batch.images],
        "messages": [[msg] for msg in batch.messages],
        "proprio_vectors": batch.proprio_data,
        "prompts": batch.prompts,
    }


def batch_inference_with_vlm(
    images_list: List[List[Image.Image]],
    messages_list: List[List[Dict[str, Any]]],
    model: Any,
    tokenizer: Any,
) -> List[str]:
    """
    Legacy function for backward compatibility.
    """
    # Flatten the nested structure for the new API
    images = [img_list[0] for img_list in images_list if img_list]
    messages = [msg_list[0] for msg_list in messages_list if msg_list]
    
    if not images or not messages:
        return []
    
    # Create a simple batch
    batch = VLMBatch(
        images=images,
        messages=messages,
        proprio_data=[],  # Not used in legacy function
        prompts=[]        # Not used in legacy function
    )
    
    # Use new inference handler
    inference_handler = VLMInference(model, tokenizer)
    return inference_handler.batch_inference(batch)
