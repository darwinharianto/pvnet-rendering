from pvnet_rendering.csrc.fps.fps_utils import farthest_point_sampling
from pvnet_rendering.opengl.opengl_renderer import OpenGLRenderer
import numpy as np

num_keypoints = 8
save_path = 'fps.txt'

renderer = OpenGLRenderer('/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/hsr_ply/fixed_hsr.ply')
sampled_kpts3d = farthest_point_sampling(pts=renderer.model['pts']/1000, sn=num_keypoints, init_center=True)
np.savetxt(save_path, sampled_kpts3d)