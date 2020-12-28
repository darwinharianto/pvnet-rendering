from typing import List
import numpy as np
from pvnet_rendering.util.blenderproc import linemod_from_blenderproc
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir

img_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/output/coco_data'
ann_path = f'{img_dir}/new_coco_annotations3.json'
K = np.array([517.799858, 0.000000, 303.876287, 0.000000, 514.807834, 238.157119, 0.000000, 0.000000, 1.000000]).reshape(3,3)
obj_positions = np.loadtxt('/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/obj_positions')
camera_positions = np.loadtxt('/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/camera_positions')
fps_3d = np.loadtxt('/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/fps.txt')
corner_3d = np.loadtxt('/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/corner3d.txt')


dst_dir = 'conversion_dst'
make_dir_if_not_exists(dst_dir)
delete_all_files_in_dir(dst_dir, ask_permission=False)

coco_dataset = COCO_Dataset.load_from_path(ann_path, img_dir=img_dir, strict=False)
coco_dataset.images.sort(attr_name='file_name')
src_img_paths = []
seg_list = []
for coco_image in coco_dataset.images:
    anns = coco_dataset.annotations.get(image_id=coco_image.id)
    assert len(anns) == 1, f"len(anns) == {len(anns)} != 1 for image_id={coco_image.id}"
    ann = anns[0]
    assert '/' not in coco_image.file_name
    src_img_paths.append(f'{img_dir}/{coco_image.file_name}')
    seg_list.append(ann.segmentation)

linemod_dataset = linemod_from_blenderproc(
    src_img_paths=src_img_paths,
    obj_positions=obj_positions,
    camera_positions=camera_positions,
    fps_3d=fps_3d,
    corner_3d=corner_3d,
    K=K,
    seg_list=seg_list,
    dst_dir=dst_dir,
    class_name='hsr'
)
linemod_dataset.save_to_path(f'{dst_dir}/output.json', overwrite=True)
coco_dataset0 = linemod_dataset.to_coco()
coco_dataset0.save_to_path(f'{dst_dir}/coco.json', overwrite=True)
# coco_dataset0.save_video(save_path='sample.avi', fps=5, show_details=True)
linemod_dataset.set_dataroot(dataroot='data/custom')
linemod_dataset.set_images_dir(img_dir='data/custom')
linemod_dataset.save_to_path(f'{dst_dir}/train.json', overwrite=True)

# Other files
from common_utils.file_utils import copy_file
np.savetxt(f'{dst_dir}/camera.txt', K)
np.savetxt(f'{dst_dir}/fps.txt', fps_3d)
np.savetxt(f'{dst_dir}/diameter.txt', np.array([1.28]))
copy_file(src_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/clayton_datasets/blackout/model.ply', dest_path=f'{dst_dir}/model.ply', silent=True)