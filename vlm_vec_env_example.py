from utils.data_utils import process_sim_data_for_vlm, batch_inference_with_vlm
from envs.vec_env import VecEnv
from unsloth import FastVisionModel
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController

# select device
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# set macros
gm.RENDER_VIEWER_CAMERA = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = False

# load VLM
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
FastVisionModel.for_inference(model)

# og vec env
num_envs = 2
vec_env = VecEnv(num_envs=num_envs, config="config/scene_config.yaml")
robot_names = [env.robots[0].name for env in vec_env.envs]

# controllers for random actions
controllers = [KeyboardRobotController(robot=env.robots[0]) for env in vec_env.envs]
# loop
while True:
    actions = [controller.get_random_action() for controller in controllers]

    # one step, get obs, and save to pkl
    obs, rewards, terminates, truncates, infos = vec_env.step(actions)
    processed_data = process_sim_data_for_vlm(obs,vec_env.envs)
    # with open("obs.pkl", "wb") as f:
    #     pickle.dump(processed_data, f)

    # process data for vlm
    inference_results = batch_inference_with_vlm(processed_data["rgb_images"],processed_data["messages"], model, tokenizer)
    # pprint(f"processed_data: {processed_data}")
    print(f"inference_results: {inference_results}")