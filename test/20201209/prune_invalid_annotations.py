from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.linemod.objects import Linemod_Dataset
from common_utils.file_utils import copy_file
from tqdm import tqdm

linemod_dataset = Linemod_Dataset.load_from_path('combined/output.json')
dataset = COCO_Dataset.load_from_path('combined/coco.json', img_dir='combined')

coco_del_ann_idx, linemod_del_ann_idx = [], []
coco_del_img_idx, linemod_del_img_idx = [], []

pbar = tqdm(total=len(dataset.images), unit='image(s)', leave=True)
for coco_image in dataset.images:
    linemod_image = linemod_dataset.images.get(file_name=coco_image.file_name)[0]
    linemod_anns = linemod_dataset.annotations.get(image_id=linemod_image.id)
    coco_anns = dataset.annotations.get(image_id=coco_image.id)
    assert len(linemod_anns) == 1
    assert len(coco_anns) == 1
    linemod_ann = linemod_anns[0]
    coco_ann = coco_anns[0]
    if coco_ann.bbox.area() <= 0 or len(coco_ann.segmentation) == 0:
        if coco_ann.bbox.area() <= 0:
            print(f'{coco_ann.id} -> coco_ann.bbox.area(): {coco_ann.bbox.area()}')
        elif len(coco_ann.segmentation) == 0:
            print(f'{coco_ann.id} -> len(coco_ann.segmentation): {len(coco_ann.segmentation)}')
        coco_del_ann_idx.append(coco_ann.id)
        linemod_del_ann_idx.append(linemod_ann.id)
        coco_del_img_idx.append(coco_ann.image_id)
        linemod_del_img_idx.append(linemod_ann.image_id)
    pbar.update()
pbar.close()
for i in list(range(len(dataset.images)))[::-1]:
    if dataset.images[i].id in coco_del_img_idx:
        del dataset.images[i]
for i in list(range(len(linemod_dataset.images)))[::-1]:
    if linemod_dataset.images[i].id in linemod_del_img_idx:
        del linemod_dataset.images[i]
for i in list(range(len(dataset.annotations)))[::-1]:
    if dataset.annotations[i].id in coco_del_ann_idx:
        del dataset.annotations[i]
for i in list(range(len(linemod_dataset.annotations)))[::-1]:
    if linemod_dataset.annotations[i].id in linemod_del_ann_idx:
        del linemod_dataset.annotations[i]

copy_file(src_path='combined/coco.json', dest_path='combined/orig_coco.json', silent=True)
copy_file(src_path='combined/output.json', dest_path='combined/orig_output.json', silent=True)

dataset.save_to_path('combined/coco.json', overwrite=True)
linemod_dataset.save_to_path('combined/output.json', overwrite=True)