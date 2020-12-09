from annotation_utils.linemod.objects import Linemod_Dataset

renders_dir = 'combined'
dataset = Linemod_Dataset.load_from_path(f'{renders_dir}/train.json')
coco_dataset = dataset.to_coco(img_dir=renders_dir, mask_dir=renders_dir, show_pbar=True)
coco_dataset.save_video(
    save_path='preview.avi', fps=5,
    show_seg=False, show_bbox=True,
    show_kpt=True, show_skeleton=False,
    show_details=True, overwrite=True
)