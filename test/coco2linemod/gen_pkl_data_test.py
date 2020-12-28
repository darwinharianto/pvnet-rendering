import numpy as np
import json
from pvnet_rendering.util.coco2linemod import project, get_RT, get_object_to_world_pose, get_world_to_camera_pose

img_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/output/coco_data'
ann_path = f'{img_dir}/new_coco_annotations3.json'

data = json.load(open(ann_path, 'r'))
annotations = data['annotations']
sample_ann = annotations[0]
# print(sample_ann.keys())
sample_keypoints = np.array(sample_ann['keypoints']).reshape(-1, 3)[:, :2]
sample_keypoints_3d = np.array(sample_ann['keypoints_3d']).reshape(-1, 4)[:, :3]
center_2d = sample_ann['center_2d']
K = np.array([517.799858, 0.000000, 303.876287, 0.000000, 514.807834, 238.157119, 0.000000, 0.000000, 1.000000]).reshape(3,3)

obj_positions = np.loadtxt('/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/obj_positions')
camera_positions = np.loadtxt('/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201223/camera_positions')

# print(f'obj_positions.shape: {obj_positions.shape}')
# print(f'obj_positions[0]: {obj_positions[0]}')
obj_pos_list, obj_rot_list = [pos[:3] for pos in obj_positions], [pos[3:] for pos in obj_positions]
# print(f'obj_pos_list[0]: {obj_pos_list[0]}')
# print(f'obj_rot_list[0]: {obj_rot_list[0]}')
sample_obj_pos, sample_obj_rot = obj_pos_list[0], obj_rot_list[0]

# print(f'camera_positions.shape: {camera_positions.shape}')
# print(f'camera_positions[0]: {camera_positions[0]}')
camera_pos_list, camera_rot_list = [pos[:3] for pos in camera_positions], [pos[3:] for pos in camera_positions]
# print(f'camera_pos_list[0]: {camera_pos_list[0]}')
# print(f'camera_rot_list[0]: {camera_rot_list[0]}')
sample_camera_pos, sample_camera_rot = camera_pos_list[0], camera_rot_list[0]

RT = get_RT(camera_pos=sample_camera_pos, camera_rot=sample_camera_rot)
# print(f'RT: {RT}')

object_to_world_pose = get_object_to_world_pose(obj_pos=sample_obj_pos, obj_rot=sample_obj_rot)
world_to_camera_pose = get_world_to_camera_pose(RT=RT, object_to_world_pose=object_to_world_pose)
# print(f'sample_keypoints_3d.shape: {sample_keypoints_3d.shape}')
# print(f'K.shape: {K.shape}')
# print(f'world_to_camera_pose.shape: {world_to_camera_pose.shape}')
calculated_keypoints = project(sample_keypoints_3d, K, world_to_camera_pose)
# print(f'calculated_keypoints: {calculated_keypoints}')
# print(f'sample_keypoints: {sample_keypoints}')

# KRT = {"K": np.asarray(K, dtype=np.float32), "RT": np.asarray(RT, dtype=np.float32)}
# print(f'KRT: {KRT}')
pose = np.asarray(RT, dtype=np.float32)
print(f'pose:\n{pose}')

# x, y = center_2d
# print(f'center_2d: {center_2d}')
# print(f'Before object_to_world_pose: {object_to_world_pose}')
# object_to_world_pose = np.array([[1, 0, 0, x],
#                                     [0, 1, 0, y],
#                                     [0, 0, 1, 0]])
# object_to_world_pose = np.append(object_to_world_pose, [[0, 0, 0, 1]], axis=0)
# print(f'After object_to_world_pose: {object_to_world_pose}')
# print(f'Before world_to_camera_pose: {world_to_camera_pose}')
# world_to_camera_pose = np.append(KRT['RT'], [[0, 0, 0, 1]], axis=0)
# world_to_camera_pose = np.dot(world_to_camera_pose, object_to_world_pose)[:3]
# print(f'After world_to_camera_pose: {world_to_camera_pose}')
# pkl_data = {'RT': world_to_camera_pose, 'K': KRT['K']}
# print(f'pkl_data:\n{pkl_data}')