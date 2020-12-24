from annotation_utils.coco.structs import COCO_Dataset

orig_dataset_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/clayton_datasets/combined'
blackout_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/clayton_datasets/offset_blackout'

dataset = COCO_Dataset.load_from_path(f'{orig_dataset_dir}/coco.json', img_dir=blackout_dir)

dataset, _ = dataset.split_into_parts(ratio=[100, len(dataset.images)-100], shuffle=True)
dataset.save_video(save_path='preview.avi', fps=5, show_details=True, show_seg=False, show_skeleton=False, show_kpt=False, show_bbox=True)