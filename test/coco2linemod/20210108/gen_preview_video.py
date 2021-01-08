from annotation_utils.coco.structs import COCO_Dataset

root_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20210108'
img_dir = f'{root_dir}/combined'
ann_path = f'{img_dir}/output.json'

dataset = COCO_Dataset.load_from_path(ann_path, img_dir=img_dir)
dataset.save_video(save_path=f'{root_dir}/combined_preview.avi', fps=5, show_details=True)