from annotation_utils.linemod.objects import Linemod_Dataset

renders1_dataset = Linemod_Dataset.load_from_path(json_path='renders1/train.json')
renders2_dataset = Linemod_Dataset.load_from_path(json_path='renders2/train.json')

combined_dataset = Linemod_Dataset.combine([renders1_dataset, renders2_dataset], show_pbar=True)
combined_dataset.move(
    dst_dataroot='combined',
    include_depth=True,
    include_RT=True,
    camera_path='renders1/camera.txt',
    fps_path='renders1/fps.txt',
    preserve_filename=False,
    use_softlink=True,
    ask_permission_on_delete=True,
    show_pbar=True
)
combined_dataset.save_to_path('combined/train.json', overwrite=True)