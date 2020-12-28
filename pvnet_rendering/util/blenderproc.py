from typing import List
import numpy as np
import cv2
from tqdm import tqdm
from mathutils import Matrix, Vector # pip install mathutils (this is blender's mathutils)
from scipy.spatial.transform import Rotation as R

from annotation_utils.linemod.objects import Linemod_Dataset, Linemod_Annotation, \
    Linemod_Image, Linemod_Category, LinemodCamera
from common_utils.file_utils import file_exists
from common_utils.path_utils import get_filename, get_dirpath_from_filepath, get_rootname_from_path
from common_utils.common_types.segmentation import Segmentation
from common_utils.common_types.point import Point2D, Point3D, Point2D_List, Point3D_List
from common_utils.common_types.angle import QuaternionList

def project(xyz: np.ndarray, K: np.ndarray, RT: np.ndarray) -> np.ndarray:
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]

    return xy

def get_3x4_RT_matrix_from_blender(location: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Use matrix_world instead to account for all constraints
    # location, rotation = matrix_world.decompose()[0:2]
    R_world2bcam = Matrix(rotation)

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((R_world2cv[0][:] + (T_world2cv[0],),
                R_world2cv[1][:] + (T_world2cv[1],),
                R_world2cv[2][:] + (T_world2cv[2],)))
    return RT

def get_RT(camera_pos: np.ndarray, camera_rot: np.ndarray) -> np.ndarray:
    RT = get_3x4_RT_matrix_from_blender(Vector(camera_pos), R.from_euler('xyz', -camera_rot).as_matrix())
    return RT

def get_object_to_world_pose(obj_pos: np.ndarray, obj_rot: np.ndarray) -> np.ndarray:
    pose_rot_obj = R.from_euler('xyz', obj_rot)
    object_to_world_pose = np.append(pose_rot_obj.as_matrix(), obj_pos.reshape(3,-1),axis=1)
    object_to_world_pose = np.append(object_to_world_pose, [[0, 0, 0, 1]], axis=0)
    return object_to_world_pose

def get_world_to_camera_pose(RT: np.ndarray, object_to_world_pose: np.ndarray) -> np.ndarray:
    world_to_camera_pose = np.append(RT, [[0, 0, 0, 1]], axis=0)
    world_to_camera_pose = np.dot(world_to_camera_pose, object_to_world_pose)[:3]
    return world_to_camera_pose

def linemod_from_blenderproc(
    src_img_paths: List[str], obj_positions: np.ndarray, camera_positions: np.ndarray,
    fps_3d: np.ndarray, corner_3d: np.ndarray, K: np.ndarray,
    seg_list: List[Segmentation],
    dst_dir: str=None, class_name: str='object',
    show_pbar: bool=True, leave_pbar: bool=True
) -> Linemod_Dataset:
    assert len(obj_positions) == len(camera_positions)
    assert len(obj_positions) == len(src_img_paths)
    assert len(obj_positions) == len(seg_list)
    center_3d = np.mean(corner_3d, axis=0)

    linemod_dataset = Linemod_Dataset()

    category_id = len(linemod_dataset.categories)
    linemod_category = Linemod_Category(
        supercategory=class_name,
        name=class_name,
        id=category_id
    )
    linemod_dataset.categories.append(linemod_category)

    pbar = tqdm(total=len(obj_positions), unit='frame(s)', leave=leave_pbar) if show_pbar else None
    if pbar is not None:
        pbar.set_description('Converting Blenderproc to Linemod')
    for obj_position, camera_position, src_img_path, seg in zip(obj_positions, camera_positions, src_img_paths, seg_list):
        assert file_exists(src_img_path), f"Couldn't find image: {src_img_path}"
        img = cv2.imread(src_img_path)
        img_h, img_w = img.shape[:2]
        if dst_dir is not None:
            dst_img_path = f'{dst_dir}/{get_filename(src_img_path)}'
            cv2.imwrite(dst_img_path, img)
        else:
            dst_img_path = src_img_path
            dst_dir = get_dirpath_from_filepath(src_img_path)
        
        image_id = len(linemod_dataset.images)
        linemod_image = Linemod_Image(
            file_name=dst_img_path,
            width=img_w, height=img_h,
            id=image_id
        )
        linemod_dataset.images.append(linemod_image)
        
        # Calculate Keypoints and Pose
        obj_pos, obj_rot = obj_position[:3], obj_position[3:]
        camera_pos, camera_rot = camera_position[:3], camera_position[3:]
        RT = get_RT(camera_pos=camera_pos, camera_rot=camera_rot)
        object_to_world_pose = get_object_to_world_pose(obj_pos=obj_pos, obj_rot=obj_rot)
        world_to_camera_pose = get_world_to_camera_pose(RT=RT, object_to_world_pose=object_to_world_pose)
        fps_2d = project(xyz=fps_3d, K=K, RT=world_to_camera_pose)
        corner_2d = project(corner_3d, K, world_to_camera_pose)
        center_2d = project(center_3d, K, world_to_camera_pose)
        pose = np.asarray(RT, dtype=np.float32)

        # Generate Mask From Segmentation
        mask = seg.draw(img=np.zeros_like(img), color=(255, 255, 255))
        mask_dst_path = f'{dst_dir}/{get_rootname_from_path(src_img_path)}_mask.png'
        cv2.imwrite(mask_dst_path, mask)

        linemod_ann = Linemod_Annotation(
            data_root=dst_dir,
            mask_path=mask_dst_path,
            type='real',
            class_name=class_name,
            corner_2d=Point2D_List.from_numpy(corner_2d, demarcation=True),
            corner_3d=Point3D_List.from_numpy(corner_3d, demarcation=True),
            center_2d=Point2D.from_numpy(center_2d.reshape(-1)),
            center_3d=Point3D.from_numpy(center_3d.reshape(-1)),
            fps_2d=Point2D_List.from_numpy(fps_2d, demarcation=True),
            fps_3d=Point3D_List.from_numpy(fps_3d, demarcation=True),
            K=LinemodCamera.from_matrix(K),
            pose=QuaternionList.from_numpy(pose),
            image_id=image_id,
            category_id=category_id,
            id=len(linemod_dataset.annotations),
            depth_path=None
        )
        linemod_dataset.annotations.append(linemod_ann)
        if pbar is not None:
            pbar.update()
    if pbar is not None:
        pbar.close()
    return linemod_dataset