from pvnet_rendering.config import cfg
from pvnet_rendering.blender.render import Renderer

cfg.BLENDER_PATH = '/home/clayton/Documents/blender-2.83.9-linux64/blender'

renderer = Renderer(
    class_type='hsr',
    bg_img_dir='/home/clayton/workspace/prj/data_keep/data/misc_dataset/COCO/train2017',
    renders_dir='renders1',
    obj_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/hsr_ply/fixed_hsr.ply',
    poses_path='poses.npy'
)
renderer.run()