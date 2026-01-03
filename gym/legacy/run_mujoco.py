import mujoco
from PIL import Image
import numpy as np
# xml = """
# <mujoco>
#   <worldbody>
# 	<light name="top" pos="0 0 1"/>

# 	<geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
# 	<geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>
# """
# # Make model and data
# model = mujoco.MjModel.from_xml_string(xml)
# data = mujoco.MjData(model)

# # Make renderer, render and show the pixels
# with mujoco.Renderer(model) as renderer:
# 	mujoco.mj_forward(model, data)

# 	renderer.update_scene(data)
# 	print(renderer.render())
# 	img_gray = Image.fromarray(renderer.render(), mode='RGB') # 'L' for grayscale

# img_gray.show()


import gymnasium as gym
env = gym.make("Hopper-v5", render_mode="rgb_array", width=1280, height=720)
print(env)
env.reset()

img_gray = Image.fromarray(env.render(), mode='RGB') # 'L' for grayscale
img_gray.show()
