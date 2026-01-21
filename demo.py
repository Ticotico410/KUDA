import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

from PIL import Image
import numpy as np
import openai

from planner.planner import KUDAPlanner
from utils import get_config_real

openai.api_key = None

config = get_config_real('configs/real_config.yaml')
planner_config = config.planner
env = None
planner = KUDAPlanner(env, planner_config)
img = Image.open('prompts/random_0.jpg')

# TODO: test
# img = Image.open('prompts/test.jpg')
img = np.array(img)[:, :, ::-1] # in BGR format

instruction = "divide the chessmen into two groups based on their color"
# instruction = "gather the purple earplugs together"
# instruction = "push the T-shaped object first to the right side and then to the green center on the image"

planner.demo_call(img, instruction)


'''
Please run this command to avoid the out of memory error:
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python demo.py
'''