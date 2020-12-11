from annotation_utils.linemod.objects import Linemod_Dataset

renders_dir = f'combined'
dataset = Linemod_Dataset.load_from_path(f'{renders_dir}/output.json')
dataset.set_images_dir('data/custom')
dataset.set_dataroot('data/custom', include_mask=True, include_depth=True)
dataset.save_to_path(f'{renders_dir}/train.json', overwrite=True)