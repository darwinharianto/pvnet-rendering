from annotation_utils.linemod.objects import Linemod_Dataset

dst_root = '/home/clayton/workspace/prj/data/misc_dataset/clayton_datasets/20201209'

datasets = [Linemod_Dataset.load_from_path(f'{dst_root}/{target}/train.json') for target in ['close_range_renders', 'mid_range_renders', 'long_range_renders']]

darwin_dataset = Linemod_Dataset.load_from_path('/home/clayton/workspace/prj/data/misc_dataset/darwin_datasets/nihonbashi2/organized/train.json')
darwin_dataset.set_dataroot('/home/clayton/workspace/prj/data/misc_dataset/darwin_datasets/nihonbashi2/organized', include_mask=True, include_depth=True)
darwin_dataset.set_images_dir('/home/clayton/workspace/prj/data/misc_dataset/darwin_datasets/nihonbashi2/organized')
datasets.append(darwin_dataset)

combined_dataset = Linemod_Dataset.combine(datasets, show_pbar=True)
combined_dataset.move(
    dst_dataroot='combined',
    include_depth=True,
    include_RT=True,
    camera_path=f'{datasets[0].annotations[0].data_root}/camera.txt',
    fps_path=f'{datasets[0].annotations[0].data_root}/fps.txt',
    preserve_filename=False,
    use_softlink=True,
    ask_permission_on_delete=True,
    show_pbar=True
)
combined_dataset.save_to_path('combined/train.json', overwrite=True)