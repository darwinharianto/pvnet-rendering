import os
import numpy as np
from PIL import Image



data_path = './data/LINEMOD/renders/'
target = 'nihonbashi'
save_pose_dir = '/home/doors/workspace/darwin/dataset/LINEMOD/nihonbashi/pose'
save_mask_dir = '/home/doors/workspace/darwin/dataset/LINEMOD/nihonbashi/mask'

def read_linemod_mask(path):
    return (np.asarray(Image.open(path))).astype(np.uint8)*255

#save pose
for asd in os.listdir(f'./data/LINEMOD/renders/{target}'):
    if asd.endswith('.pkl'):
        tes = np.load(os.path.join(data_path, target, asd), allow_pickle=True)
        iterator = asd.split('_')[0]
        np.save(os.path.join(save_pose_dir, 'pose'+iterator), tes['RT'])
        
    elif asd.endswith('.png'):
        filename = os.path.join(data_path, target, asd)
        mask_image = read_linemod_mask(filename)

        iterator = asd.split('_')[0]
        im = Image.fromarray(mask_image)
        im.save(os.path.join(save_mask_dir, iterator+'.png'))
