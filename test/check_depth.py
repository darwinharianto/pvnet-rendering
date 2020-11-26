import cv2
import numpy as np
from PIL import Image
from common_utils.path_utils import get_all_files_of_extension
from streamer.cv_viewer import cv_simple_image_viewer

renders_dir = 'renders0/hsr'

depth_paths = get_all_files_of_extension(renders_dir, extension='png')
depth_paths.sort()

def read_linemod_mask(path):
    return (np.asarray(Image.open(path))).astype(np.uint8)*255

for depth_path in depth_paths:
    depth_img = read_linemod_mask(depth_path)
    print(f'depth_img.shape: {depth_img.shape}')
    print(depth_img.max())
    quit_flag = cv_simple_image_viewer(depth_img, preview_width=1000)
    if quit_flag:
        break