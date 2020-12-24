from tqdm import tqdm
from common_utils.path_utils import rel_to_abs_path, get_filename, \
    get_rootname_from_filename, get_filename
from common_utils.file_utils import make_dir_if_not_exists, delete_dir_if_exists, create_softlink, \
    file_exists, copy_file
from annotation_utils.linemod.objects import Linemod_Dataset
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.common_types.bbox import BBox
import cv2
import numpy as np
import random

def offset_value(val: float, offset_range: (float, float)) -> float:
    assert offset_range[0] <= offset_range[1]
    offset = offset_range[0] + (offset_range[1] - offset_range[0]) * random.random()
    return val + offset

def offset_bbox(bbox: BBox, offset_w_prop: float, offset_h_prop: float, frame_shape: tuple) -> BBox:
    new_xmin = offset_value(bbox.xmin, offset_range=(-1*offset_w_prop*bbox.width, offset_w_prop*bbox.width))
    new_xmax = offset_value(bbox.xmax, offset_range=(-1*offset_w_prop*bbox.width, offset_w_prop*bbox.width))
    new_ymin = offset_value(bbox.ymin, offset_range=(-1*offset_h_prop*bbox.height, offset_h_prop*bbox.height))
    new_ymax = offset_value(bbox.ymax, offset_range=(-1*offset_h_prop*bbox.height, offset_h_prop*bbox.height))
    new_bbox = BBox(xmin=new_xmin, ymin=new_ymin, xmax=new_xmax, ymax=new_ymax)
    new_bbox = new_bbox.clip_at_bounds(frame_shape=frame_shape)
    assert new_bbox.width > 0.0, f'new_bbox.width: {new_bbox.width} <= 0.0'
    assert new_bbox.height > 0.0, f'new_bbox.height: {new_bbox.height} <= 0.0'
    return new_bbox

orig_dataset_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/clayton_datasets/combined'

linemod_dataset = Linemod_Dataset.load_from_path(f'{orig_dataset_dir}/output.json')
dataset = COCO_Dataset.load_from_path(f'{orig_dataset_dir}/coco.json', img_dir=orig_dataset_dir)
dst_dir = rel_to_abs_path(f'{orig_dataset_dir}/../offset_blackout')
delete_dir_if_exists(dst_dir)
make_dir_if_not_exists(dst_dir)

pbar = tqdm(total=len(dataset.images), unit='image(s)')
pbar.set_description('Linking Files...')
for coco_image in dataset.images:
    linemod_image = linemod_dataset.images.get(file_name=coco_image.file_name)[0]
    linemod_anns = linemod_dataset.annotations.get(image_id=linemod_image.id)
    coco_anns = dataset.annotations.get(image_id=coco_image.id)
    assert len(linemod_anns) == 1
    assert len(coco_anns) == 1
    linemod_ann = linemod_anns[0]
    coco_ann = coco_anns[0]

    # Link Images
    src_img_path = coco_image.coco_url
    assert file_exists(src_img_path), f'File not found: {src_img_path}'
    dst_img_path = rel_to_abs_path(f'{dst_dir}/{get_filename(src_img_path)}')
    
    img = cv2.imread(src_img_path)
    bbox = coco_ann.bbox
    bbox = offset_bbox(bbox=bbox, offset_w_prop=0.2, offset_h_prop=0.2, frame_shape=img.shape)
    result = bbox.crop_and_paste(src_img=img, dst_img=np.zeros_like(img))
    cv2.imwrite(dst_img_path, result)

    # Link Masks
    src_mask_path = rel_to_abs_path(f'{orig_dataset_dir}/{get_filename(linemod_ann.mask_path)}')
    assert file_exists(src_mask_path), f'File not found: {src_mask_path}'
    mask = cv2.imread(src_mask_path)
    mask = bbox.crop_and_paste(src_img=mask, dst_img=np.zeros_like(mask))
    dst_mask_path = rel_to_abs_path(f'{dst_dir}/{get_filename(linemod_ann.mask_path)}')
    # copy_file(src_path=src_mask_path, dest_path=dst_mask_path, silent=True)
    cv2.imwrite(dst_mask_path, mask)
    pbar.update()
pbar.close()

# Fix data_root
for ann in dataset.annotations:
    ann.data_root = 'data/custom'

# Link Other
copy_file(src_path=rel_to_abs_path(f'{orig_dataset_dir}/train.json'), dest_path=rel_to_abs_path(f'{dst_dir}/train.json'), silent=True)
copy_file(src_path=rel_to_abs_path(f'{orig_dataset_dir}/camera.txt'), dest_path=rel_to_abs_path(f'{dst_dir}/camera.txt'), silent=True)
copy_file(src_path=rel_to_abs_path(f'{orig_dataset_dir}/diameter.txt'), dest_path=rel_to_abs_path(f'{dst_dir}/diameter.txt'), silent=True)
copy_file(src_path=rel_to_abs_path(f'{orig_dataset_dir}/fps.txt'), dest_path=rel_to_abs_path(f'{dst_dir}/fps.txt'), silent=True)