import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

import time
import openai

from envs.real_env import RealEnv
from planner.planner import KUDAPlanner
from utils import get_config_real

openai.api_key = None


config = get_config_real('configs/real_config.yaml')
env_config = config.env
env = RealEnv(env_config)
env.start(exposure_time=3)
time.sleep(3)

print(env.serial_numbers)

planner_config = config.planner
planner = KUDAPlanner(env, planner_config)
# object = material = 'rope'
object = material = 'cube'
# object = material = 'T_shape'

# object = 'coffee_beans'
# object = 'candy'
# material = 'granular'

# instruction = 'straighten the rope'
# instruction = 'make the rope into a "V" shape'
# instruction = 'put two ends of the rope together'
# instruction = 'move all the cubes to the pink cross'
# instruction = 'move the yellow cube to the red cross'
# instruction = 'move all the coffee beans to the red cross'
# instruction = "move the orange T into the pink square"
# instruction = "move the blue T into the white square"
# instruction = "move all the red cubes together"

instruction = "move the red cube to the white square"


planner(object, material, instruction)


'''
Please run this command to avoid the out of memory error:
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python launch.py
'''