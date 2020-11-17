import os
from plyfile import PlyData
import numpy as np
from ..csrc.fps import fps_utils
from ..opengl.opengl_renderer import OpenGLRenderer
import tqdm
from PIL import Image
from . import base_utils
import json

def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points


def sample_fps_points(data_root):
    ply_path = os.path.join(data_root, 'model.ply')
    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def ndds_record_ann(model_meta, img_id, ann_id, images, annotations, cls_type):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    length = int(input("How many pictures do you want to use? "))
    inds = range(length)

    for ind in tqdm.tqdm(inds):
        desired_class = cls_type # name of the objectclass, on which you want to train

        number = str(ind) 
        number = number.zfill(6)

        # getting rgb
        datei = number + '.png'
        rgb_path = os.path.join(data_root, datei)

        rgb = Image.open(rgb_path).convert('RGB')
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        # path to annotations from ndds
        datei = number + '.json'
        pose_path = os.path.join(data_root, datei)


        # getting pose of annotations from ndds

        with open(pose_path,'r') as file:
            annotation = json.loads(file.read())
        
        object_from_annotation = annotation['objects']
        object_class = object_from_annotation[0]["class"]
       
        if desired_class in object_class:

            # translation
            translation = np.array(object_from_annotation[0]['location']) * 10

            # rotation
            rotation = np.asarray(object_from_annotation[0]['pose_transform'])[0:3, 0:3]
            rotation = np.dot(rotation, np.array([[-1, 0, 0],[0, -1, 0],[0, 0, -1]]))
            rotation = np.dot(rotation.T, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))

            # pose
            pose = np.column_stack((rotation, translation))
        else:
            print("No such class in annotations!")
            pass 

        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)[0]

        fps_2d = base_utils.project(fps_3d, K, pose)

        # getting segmentation-mask
        datei = number + '.cs.png'
        mask_path = os.path.join(data_root, datei)

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})

        anno.update({'data_root': data_root})

        anno.update({'type': 'real', 'cls': cls_type})
        annotations.append(anno)

    return img_id, ann_id


from annotation_utils.linemod.objects import Linemod_Dataset, Linemod_Annotation, \
    LinemodCamera, Linemod_Image, Linemod_Category
from common_utils.path_utils import get_all_files_of_extension, get_rootname_from_path, get_filename
from common_utils.file_utils import file_exists
from common_utils.common_types.point import Point2D_List, Point3D_List, Point2D, Point3D
from common_utils.common_types.angle import QuaternionList
import pickle
from PIL import Image

def read_linemod_mask(path) -> np.ndarray:
    return (np.asarray(Image.open(path))).astype(np.uint8)*255

def custom_to_coco(data_root, ply_path: str, camera_load_path: str, fps_load_path: str, class_name: str, ann_save_path: str=None, show_pbar: bool=False):
    renderer = OpenGLRenderer(ply_path)
    K = np.loadtxt(camera_load_path)

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.loadtxt(fps_load_path)

    dataset = Linemod_Dataset()

    category_id = len(dataset.categories)
    linemod_category = Linemod_Category(supercategory=None, name=class_name, id=category_id)
    dataset.categories.append(linemod_category)

    img_paths = get_all_files_of_extension(data_root, extension='jpg')
    img_paths.sort()
    pbar = tqdm.tqdm(total=len(img_paths), unit='image(s)') if show_pbar else None
    for img_path in img_paths:
        img_filename = get_filename(img_path)
        if pbar is not None:
            pbar.set_description(img_filename)

        rootname = get_rootname_from_path(img_path)
        depth_path = f'{data_root}/{rootname}_depth.png'
        assert file_exists(depth_path)
        pkl_path = f'{data_root}/{rootname}_RT.pkl'
        assert file_exists(pkl_path)
        pkl_data = pickle.load(open(pkl_path, 'rb'))
        RT = pkl_data['RT']
        K = pkl_data['K']

        mask_img = read_linemod_mask(depth_path)
        mask_path = f'{data_root}/{rootname}_mask.png'
        Image.fromarray(mask_img).save(mask_path)

        corner_2d = base_utils.project(corner_3d, K, RT)
        center_2d = base_utils.project(center_3d[None], K, RT)[0]
        fps_2d = base_utils.project(fps_3d, K, RT)

        img_h, img_w = mask_img.shape[:2]
        image_id = len(dataset.images)
        linemod_image = Linemod_Image(
            file_name=img_filename,
            width=img_w, height=img_h,
            id=image_id
        )
        dataset.images.append(linemod_image)

        linemod_ann = Linemod_Annotation(
            data_root=data_root,
            mask_path=mask_path,
            type='real',
            class_name=class_name,
            corner_2d=Point2D_List.from_numpy(corner_2d, demarcation=True),
            corner_3d=Point3D_List.from_numpy(corner_3d, demarcation=True),
            center_2d=Point2D.from_numpy(center_2d),
            center_3d=Point3D.from_numpy(center_3d),
            fps_2d=Point2D_List.from_numpy(fps_2d, demarcation=True),
            fps_3d=Point3D_List.from_numpy(fps_3d, demarcation=True),
            K=LinemodCamera.from_matrix(K),
            pose=QuaternionList.from_list(RT.tolist()),
            image_id=image_id,
            category_id=category_id,
            id=len(dataset.annotations),
            depth_path=None
        )
        dataset.annotations.append(linemod_ann)
        if pbar is not None:
            pbar.update()

    dataset.save_to_path(
        save_path=ann_save_path if ann_save_path is not None else f'{data_root}/train.json',
        overwrite=True
    )
    pbar.close()