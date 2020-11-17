import math
from pvnet_rendering.blender.dataset_maker import DatasetMaker
from pvnet_rendering.config.loadable_config import PVNet_Config

worker = DatasetMaker(
    dst_dir='renders1',
    cfg=PVNet_Config(
        blender_path='/home/clayton/Documents/blender-2.83.9-linux64/blender',
        num_syn=10,
        width=640, height=480
    )
)
worker.run(
    ply_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/hsr_ply/orig_hsr.ply',
    material_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/hsr_ply/hsr_material.001.jpg',
    bg_img_dir='/home/clayton/workspace/prj/data_keep/data/misc_dataset/COCO/train2017',
    class_name='hsr',
    num_keypoints=8,
    r_range=(1, 10),
    yaw_range=(-math.pi, math.pi)
)