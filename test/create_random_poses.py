import math
import random
import numpy as np
from pvnet_rendering.config import cfg
from pvnet_rendering.util.pose import get_random_pose

poses = []

for i in range(cfg.NUM_SYN):
    pose = get_random_pose(
        r_range=(1.5,4),
        theta_range=(-math.pi/36, math.pi/36),
        azi_range=(-math.pi/36, math.pi/36),
        roll_range=(-math.pi/36, math.pi/36),
        pitch_range=(-math.pi/36, math.pi/36),
        yaw_range=(-math.pi, math.pi)
    )
    poses.append(pose)
poses = np.array(poses)
np.save('poses.npy', poses)

# yaw (about y-axis)
# pitch (about x-axis)
# roll (about z-axis)
# I feel like the ranges still don't match up 💦 I will have to figure out what's going on later.