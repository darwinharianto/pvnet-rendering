from pvnet_rendering.config import cfg
from pvnet_rendering.blender.render import Renderer

cfg.BLENDER_PATH = '/home/clayton/Documents/blender-2.83.9-linux64/blender'

renderer = Renderer(
    class_type='hsr',
    bg_img_dir='/home/clayton/workspace/prj/data_keep/data/misc_dataset/COCO/train2017',
    data_dir='data0',
    renders_dir='renders0',
    obj_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/hsr_ply/orig_hsr.ply'
)
renderer.run()