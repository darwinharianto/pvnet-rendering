import math
import random
import numpy as np
from pvnet_rendering.config import cfg

def get_random_translation(r_range: (float, float)=(1, 10), theta_range: (float, float)=(-math.pi/3, math.pi/3), azi_range: (float, float)=(-math.pi/3, math.pi/3)) -> [float, float, float]:
    r = random.random() * (r_range[1] - r_range[0]) + r_range[0]
    theta = random.random() * (theta_range[1] - theta_range[0]) + theta_range[0]
    azi = random.random() * (azi_range[1] - azi_range[0]) + azi_range[0]

    x = r * math.sin(theta) * math.cos(azi)
    y = r * math.sin(theta) * math.sin(azi)
    z = r * math.cos(theta)
    return [x, y, z]

def get_random_pose(roll_range: (float, float)=(-math.pi, math.pi), pitch_range: (float, float)=(-math.pi, math.pi), yaw_range: (float, float)=(-math.pi, math.pi)) -> [float, float, float]:
    roll = random.random() * (roll_range[1] - roll_range[0]) + roll_range[0]
    pitch = random.random() * (pitch_range[1] - pitch_range[0]) + pitch_range[0]
    yaw = random.random() * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    return [roll, pitch, yaw]

poses = []

for i in range(cfg.NUM_SYN):
    x, y, z = get_random_translation(r_range=(1.5,4), theta_range=(-math.pi/36, math.pi/36), azi_range=(-math.pi/36, math.pi/36))
    roll, pitch, yaw = get_random_pose(roll_range=(-math.pi/36, math.pi/36), pitch_range=(-math.pi/36, math.pi/36), yaw_range=(-math.pi, math.pi))
    pose = [yaw*180/math.pi, pitch*180/math.pi, roll*180/math.pi, x, y, z]
    poses.append(pose)
poses = np.array(poses)
np.save('poses.npy', poses)

# yaw (about y-axis)
# pitch (about x-axis)
# roll (about z-axis)
# I feel like the ranges still don't match up 💦 I will have to figure out what's going on later.