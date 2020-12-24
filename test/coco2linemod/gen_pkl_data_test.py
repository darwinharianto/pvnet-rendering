import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.linemod.objects import Linemod_Dataset

from mathutils import Matrix, Vector # pip install mathutils (this is blender's mathutils)

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]

    return xy

def get_3x4_RT_matrix_from_blender(location, rotation):
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

img_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/output/coco_data'
ann_path = f'{img_dir}/new_coco_annotations3.json'

data = json.load(open(ann_path, 'r'))
annotations = data['annotations']
sample_ann = annotations[0]
print(sample_ann.keys())
sample_keypoints = np.array(sample_ann['keypoints']).reshape(-1, 3)[:, :2]
sample_keypoints_3d = np.array(sample_ann['keypoints_3d']).reshape(-1, 4)[:, :3]
center_2d = sample_ann['center_2d']
K = np.array([517.799858, 0.000000, 303.876287, 0.000000, 514.807834, 238.157119, 0.000000, 0.000000, 1.000000]).reshape(3,3)

obj_positions = np.loadtxt('/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/obj_positions')
camera_positions = np.loadtxt('/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/camera_positions')

print(f'obj_positions.shape: {obj_positions.shape}')
print(f'obj_positions[0]: {obj_positions[0]}')
obj_pos_list, obj_rot_list = [pos[:3] for pos in obj_positions], [pos[3:] for pos in obj_positions]
print(f'obj_pos_list[0]: {obj_pos_list[0]}')
print(f'obj_rot_list[0]: {obj_rot_list[0]}')
sample_obj_pos, sample_obj_rot = obj_pos_list[0], obj_rot_list[0]

print(f'camera_positions.shape: {camera_positions.shape}')
print(f'camera_positions[0]: {camera_positions[0]}')
camera_pos_list, camera_rot_list = [pos[:3] for pos in camera_positions], [pos[3:] for pos in camera_positions]
print(f'camera_pos_list[0]: {camera_pos_list[0]}')
print(f'camera_rot_list[0]: {camera_rot_list[0]}')
sample_camera_pos, sample_camera_rot = camera_pos_list[0], camera_rot_list[0]

pose_trans_obj = sample_obj_pos
pose_rot_obj = R.from_euler('xyz', sample_obj_rot)
object_to_world_pose = np.append(pose_rot_obj.as_matrix(), pose_trans_obj.reshape(3,-1),axis=1)
object_to_world_pose = np.append(object_to_world_pose, [[0, 0, 0, 1]], axis=0)

RT = get_3x4_RT_matrix_from_blender(Vector(sample_camera_pos), R.from_euler('xyz', -sample_camera_rot).as_matrix())
print(f'RT: {RT}')

world_to_camera_pose = np.append(RT, [[0, 0, 0, 1]], axis=0)
world_to_camera_pose = np.dot(world_to_camera_pose, object_to_world_pose)[:3]
print(f'sample_keypoints_3d.shape: {sample_keypoints_3d.shape}')
print(f'K.shape: {K.shape}')
print(f'world_to_camera_pose.shape: {world_to_camera_pose.shape}')
calculated_keypoints = project(sample_keypoints_3d, K, world_to_camera_pose)
print(f'calculated_keypoints: {calculated_keypoints}')
print(f'sample_keypoints: {sample_keypoints}')

KRT = {"K": np.asarray(K, dtype=np.float32), "RT": np.asarray(RT, dtype=np.float32)}
print(f'KRT: {KRT}')

x, y = center_2d
object_to_world_pose = np.array([[1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, 0]])
object_to_world_pose = np.append(object_to_world_pose, [[0, 0, 0, 1]], axis=0)
world_to_camera_pose = np.append(KRT['RT'], [[0, 0, 0, 1]], axis=0)
world_to_camera_pose = np.dot(world_to_camera_pose, object_to_world_pose)[:3]
pkl_data = {'RT': world_to_camera_pose, 'K': KRT['K']}
print(f'pkl_data:\n{pkl_data}')