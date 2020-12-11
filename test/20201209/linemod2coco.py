from annotation_utils.linemod.objects import Linemod_Dataset

renders_dir = f'combined'
dataset = Linemod_Dataset.load_from_path(f'{renders_dir}/output.json')
coco_dataset = dataset.to_coco(img_dir=renders_dir, mask_dir=renders_dir, show_pbar=True)
coco_dataset.save_to_path(f'{renders_dir}/coco.json', overwrite=True)

num_images = len(coco_dataset.images)
sample_dataset, _ = coco_dataset.split_into_parts(ratio=[100, num_images-100], shuffle=True)
sample_dataset.save_video(
    save_path='sample.avi', fps=5,
    show_seg=False, show_bbox=True,
    show_kpt=True, show_skeleton=False,
    show_details=True, overwrite=True
)