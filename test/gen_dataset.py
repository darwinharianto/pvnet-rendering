from typing import cast
from pvnet_rendering.config import cfg
from pvnet_rendering.blender.render import Renderer
from common_utils.file_utils import delete_dir_if_exists, make_dir_if_not_exists

raise NotImplementedError('Still a work in progress.')

class DatasetMaker:
    def __init__(self, dst_dir: str, blender_path: str):
        self.dst_dir = dst_dir
        cfg.BLENDER_PATH = blender_path
        self.renderer = cast(Renderer, None)

    def init_dst_dir(self):
        delete_dir_if_exists(self.dst_dir)
        make_dir_if_not_exists(self.dst_dir)
    
    def run(self):
        self.init_dst_dir()

cfg.BLENDER_PATH = '/home/clayton/Documents/blender-2.83.9-linux64/blender'

renderer = Renderer(
    class_type='hsr',
    bg_img_dir='/home/clayton/workspace/prj/data_keep/data/misc_dataset/COCO/train2017',
    renders_dir='renders1',
    obj_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/hsr_ply/fixed_hsr.ply',
    poses_path='poses.npy'
)
renderer.run()