import math
import numpy as np
from typing import cast
from tqdm import tqdm
from ..config.loadable_config import PVNet_Config
from .render import Renderer
from ..opengl.opengl_renderer import OpenGLRenderer
from ..csrc.fps.fps_utils import farthest_point_sampling
from ..util.pose import get_random_pose
from ..util.handle_custom_dataset import custom_to_coco
from common_utils.file_utils import delete_dir_if_exists, make_dir_if_not_exists
from annotation_utils.linemod.objects import LinemodCamera

class DatasetMaker:
    def __init__(self, dst_dir: str, cfg: PVNet_Config):
        self.dst_dir = dst_dir
        self.cfg = cfg
        self.renderer = cast(Renderer, None)
        self.camera_path = cast(str, None)
        self.fps_path = cast(str, None)
        self.poses_path = cast(str, None)
        self.cfg_path = cast(str, None)

    def init_dst_dir(self):
        delete_dir_if_exists(self.dst_dir)
        make_dir_if_not_exists(self.dst_dir)
    
    def init_camera(self):
        camera_path = f'{self.dst_dir}/camera.txt'
        camera = LinemodCamera.from_matrix(self.cfg.K)
        camera.save_to_txt(camera_path, overwrite=True)
        self.camera_path = camera_path

    def init_kpts(self, ply_path: str, num_keypoints: int=8, scale: float=0.001):
        worker = OpenGLRenderer(ply_path)
        sampled_kpts3d = farthest_point_sampling(pts=worker.model['pts']*scale, sn=num_keypoints, init_center=True)
        fps_path = f'{self.dst_dir}/fps.txt'
        np.savetxt(fps_path, sampled_kpts3d)
        self.fps_path = fps_path

    def init_poses(
        self,
        r_range: (float, float)=(1, 10),
        theta_range: (float, float)=(-math.pi/3, math.pi/3),
        azi_range: (float, float)=(-math.pi/3, math.pi/3),
        roll_range: (float, float)=(-math.pi, math.pi),
        pitch_range: (float, float)=(-math.pi, math.pi),
        yaw_range: (float, float)=(-math.pi, math.pi),
        x_offset: float=None, y_offset: float=None, z_offset: float=None
    ):
        poses = []
        for i in range(self.cfg.num_syn):
            pose = get_random_pose(
                r_range=r_range,
                theta_range=theta_range,
                azi_range=azi_range,
                roll_range=roll_range,
                pitch_range=pitch_range,
                yaw_range=yaw_range,
                x_offset=x_offset,
                y_offset=y_offset,
                z_offset=z_offset
            )
            poses.append(pose)
        poses = np.array(poses)
        poses_path = f'{self.dst_dir}/poses.npy'
        np.save(poses_path, poses)
        self.poses_path = poses_path

    def init_renderer(self, bg_img_dir: str, ply_path: str, material_path: str):
        assert self.poses_path is not None, f'{type(self).__name__}.init_poses() must be called before {type(self).__name__}.init_renderer()'
        self.renderer = Renderer(
            bg_img_dir=bg_img_dir,
            renders_dir=self.dst_dir,
            obj_path=ply_path,
            material_path=material_path,
            poses_path=self.poses_path,
            cfg=self.cfg
        )

    def init_config(self):
        cfg_path = f'{self.dst_dir}/config.json'
        self.cfg.save_to_path(cfg_path, overwrite=True)
        self.cfg_path = cfg_path

    def run(
        self,
        ply_path: str, material_path: str, bg_img_dir: str, class_name: str,
        num_keypoints: int=8, scale: float=0.001,
        r_range: (float, float)=(1, 10),
        theta_range: (float, float)=(-math.pi/36, math.pi/36),
        azi_range: (float, float)=(-math.pi/36, math.pi/36),
        roll_range: (float, float)=(-math.pi/36, math.pi/36),
        pitch_range: (float, float)=(-math.pi/36, math.pi/36),
        yaw_range: (float, float)=(-math.pi/36, math.pi/36),
        x_offset: float=None, y_offset: float=None, z_offset: float=None,
        show_init_pbar: bool=True
    ):
        init_pbar = tqdm(total=6, unit='step(s)') if show_init_pbar else None
        if init_pbar is not None:
            init_pbar.set_description('Preparing Destination Directory')
        self.init_dst_dir()
        if init_pbar is not None:
            init_pbar.update()
            init_pbar.set_description('Initializing Camera')
        self.init_camera()
        if init_pbar is not None:
            init_pbar.update()
            init_pbar.set_description('Initializing Keypoints')
        self.init_kpts(ply_path=ply_path, num_keypoints=num_keypoints, scale=scale)
        if init_pbar is not None:
            init_pbar.update()
            init_pbar.set_description('Initializing Poses')
        self.init_poses(
            r_range=r_range,
            theta_range=theta_range,
            azi_range=azi_range,
            roll_range=roll_range,
            pitch_range=pitch_range,
            yaw_range=yaw_range,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset
        )
        if init_pbar is not None:
            init_pbar.update()
            init_pbar.set_description('Initializing Renderer')
        self.init_renderer(bg_img_dir=bg_img_dir, ply_path=ply_path, material_path=material_path)
        if init_pbar is not None:
            init_pbar.update()
            init_pbar.set_description('Saving Configuration')
        self.init_config()
        if init_pbar is not None:
            init_pbar.update()
            init_pbar.close()
        self.renderer.run(cfg_path=self.cfg_path)
        custom_to_coco(
            data_root=self.dst_dir,
            ply_path=ply_path,
            camera_load_path=self.camera_path,
            fps_load_path=self.fps_path,
            class_name=class_name,
            ann_save_path=f'{self.dst_dir}/train.json',
            show_pbar=True
        )