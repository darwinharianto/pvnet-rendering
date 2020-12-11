from tqdm import tqdm
from common_utils.path_utils import rel_to_abs_path, get_filename, \
    get_rootname_from_filename, get_filename
from common_utils.file_utils import make_dir_if_not_exists, delete_dir_if_exists, create_softlink, \
    file_exists
from annotation_utils.linemod.objects import Linemod_Dataset
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.common_types.bbox import BBox
import cv2
import numpy as np

linemod_dataset = Linemod_Dataset.load_from_path('combined/output.json')
dataset = COCO_Dataset.load_from_path('combined/coco.json', img_dir='combined')
dst_dir = rel_to_abs_path('blackout')
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
    bbox = bbox.clip_at_bounds(frame_shape=img.shape)
    result = bbox.crop_and_paste(src_img=img, dst_img=np.zeros_like(img))
    cv2.imwrite(dst_img_path, result)

    # Link Masks
    src_mask_path = rel_to_abs_path(linemod_ann.mask_path)
    dst_mask_path = rel_to_abs_path(f'{dst_dir}/{get_filename(linemod_ann.mask_path)}')
    create_softlink(src_path=src_mask_path, dst_path=dst_mask_path)
    pbar.update()
pbar.close()

# Fix data_root
for ann in dataset.annotations:
    ann.data_root = 'data/custom'

# Link Other
create_softlink(src_path=rel_to_abs_path(f'combined/train.json'), dst_path=rel_to_abs_path(f'{dst_dir}/train.json'))
create_softlink(src_path=rel_to_abs_path('combined/camera.txt'), dst_path=rel_to_abs_path(f'{dst_dir}/camera.txt'))
create_softlink(src_path=rel_to_abs_path('combined/diameter.txt'), dst_path=rel_to_abs_path(f'{dst_dir}/diameter.txt'))
create_softlink(src_path=rel_to_abs_path('combined/fps.txt'), dst_path=rel_to_abs_path(f'{dst_dir}/fps.txt'))