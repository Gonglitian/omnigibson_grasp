import torch
import numpy as np
import omnigibson as og
# Load model directly
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Prepare the input data
# Assuming 'proprio' is a tensor of shape (n,) and 'rgb' is a tensor of shape (H, W, 4)
# Replace the following with your actual data
proprio_tensor = torch.tensor([...])  # Replace with your proprioceptive data
rgb_tensor = torch.tensor([...], dtype=torch.uint8)  # Replace with your RGB image data

# Convert the RGB tensor to a PIL Image
from PIL import Image
import numpy as np

# Remove the alpha channel if present
if rgb_tensor.shape[2] == 4:
    rgb_tensor = rgb_tensor[:, :, :3]

# Convert to PIL Image
image = Image.fromarray(rgb_tensor.numpy())

# Define the input message
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Based on the robot's proprioceptive data, determine the next action."}
        ],
    }
]

# Apply the chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process vision information
image_inputs, video_inputs = process_vision_info(messages)

# Prepare the inputs for the model
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

# Move inputs to the appropriate device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Perform inference
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# Decode the generated output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Print the output
print(output_text[0])
