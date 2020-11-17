from pvnet_rendering.util.handle_custom_dataset import custom_to_coco

data_dir = 'data0'
renders_dir = 'renders0'

custom_to_coco(
    data_root=f'{renders_dir}/hsr',
    ply_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/hsr_ply/orig_hsr.ply',
    camera_load_path='camera.txt',
    fps_load_path='fps.txt',
    class_name='hsr',
    ann_save_path=f'{data_dir}/train.json',
    show_pbar=True
)

# TODO: Make a class for random pose gen + camera gen + fps gen + render + convert