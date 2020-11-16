from streamer.recorder import Recorder
from common_utils.path_utils import get_all_files_of_extension
import cv2
from typing import cast
from tqdm import tqdm

preview_save_path = 'preview.avi'
renders_dir = 'renders1/hsr'
img_paths = get_all_files_of_extension(renders_dir, extension='jpg')
img_paths.sort()

recorder = cast(Recorder, None)

pbar = tqdm(total=len(img_paths), unit='image(s)')
for img_path in img_paths:
    img = cv2.imread(img_path)

    if recorder is None:
        img_h, img_w = img.shape[:2]
        recorder = Recorder(output_path=preview_save_path, output_dims=(img_w, img_h), fps=5)
    recorder.write(img)
    pbar.update()
recorder.close()