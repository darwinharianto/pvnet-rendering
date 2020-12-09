import random
import math

def get_random_translation(
    r_range: (float, float)=(1, 10),
    theta_range: (float, float)=(-math.pi/3, math.pi/3),
    azi_range: (float, float)=(-math.pi/3, math.pi/3)
) -> [float, float, float]:
    r = random.random() * (r_range[1] - r_range[0]) + r_range[0]
    theta = random.random() * (theta_range[1] - theta_range[0]) + theta_range[0]
    azi = random.random() * (azi_range[1] - azi_range[0]) + azi_range[0]

    x = r * math.sin(theta) * math.cos(azi)
    y = r * math.sin(theta) * math.sin(azi)
    z = r * math.cos(theta)
    return [x, y, z]

def get_random_rotation(
    roll_range: (float, float)=(-math.pi, math.pi),
    pitch_range: (float, float)=(-math.pi, math.pi),
    yaw_range: (float, float)=(-math.pi, math.pi)
) -> [float, float, float]:
    roll = random.random() * (roll_range[1] - roll_range[0]) + roll_range[0]
    pitch = random.random() * (pitch_range[1] - pitch_range[0]) + pitch_range[0]
    yaw = random.random() * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    return [roll, pitch, yaw]

def get_random_pose(
    r_range: (float, float)=(1, 10),
    theta_range: (float, float)=(-math.pi/3, math.pi/3),
    azi_range: (float, float)=(-math.pi/3, math.pi/3),
    roll_range: (float, float)=(-math.pi, math.pi),
    pitch_range: (float, float)=(-math.pi, math.pi),
    yaw_range: (float, float)=(-math.pi, math.pi),
    x_offset: float=None, y_offset: float=None, z_offset: float=None
):
    x, y, z = get_random_translation(r_range=r_range, theta_range=theta_range, azi_range=azi_range)
    if x_offset is not None:
        if isinstance(x_offset, float):
            x += x_offset
        elif isinstance(x_offset, (list, tuple)) and all([type(val) in [float, int] for val in x_offset]):
            x_offset0 = random.random() * (x_offset[1] - x_offset[0]) + x_offset[0]
            x += x_offset0
        else:
            raise TypeError(f'Invalid type(x_offset): {type(x_offset)}')
    if y_offset is not None:
        if isinstance(y_offset, float):
            y += y_offset
        elif isinstance(y_offset, (list, tuple)) and all([type(val) in [float, int] for val in y_offset]):
            y_offset0 = random.random() * (y_offset[1] - y_offset[0]) + y_offset[0]
            y += y_offset0
        else:
            raise TypeError(f'Invalid type(y_offset): {type(y_offset)}')
    if z_offset is not None:
        if isinstance(z_offset, float):
            z += z_offset
        elif isinstance(z_offset, (list, tuple)) and all([type(val) in [float, int] for val in z_offset]):
            z_offset0 = random.random() * (z_offset[1] - z_offset[0]) + z_offset[0]
            z += z_offset0
        else:
            raise TypeError(f'Invalid type(z_offset): {type(z_offset)}')
    roll, pitch, yaw = get_random_rotation(roll_range=roll_range, pitch_range=pitch_range, yaw_range=yaw_range)
    return [yaw*180/math.pi, pitch*180/math.pi, roll*180/math.pi, x, y, z]