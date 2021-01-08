from annotation_utils.coco.structs import COCO_Dataset

root_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20210108'
targets = [
    'output_1',
    'output_2',
    'output_3',
    'output_4',
    'output_5',
]
target_img_dirs = [f'{root_dir}/{target}/coco_data' for target in targets]
target_ann_paths = [f'{target_img_dir}/new_coco_annotations3.json' for target_img_dir in target_img_dirs]

datasets = [COCO_Dataset.load_from_path(ann_path, img_dir=img_dir) for img_dir, ann_path in zip(target_img_dirs, target_ann_paths)]
combined_dataset = COCO_Dataset.combine(datasets, show_pbar=True)
combined_dataset.move_images(f'{root_dir}/combined', update_img_paths=True, show_pbar=True, overwrite=True)
combined_dataset.save_to_path(f'{root_dir}/combined/output.json', overwrite=True)