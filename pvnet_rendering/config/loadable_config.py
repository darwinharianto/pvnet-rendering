import os
import numpy as np
from common_utils.base.basic import BasicLoadableObject
from common_utils.file_utils import dir_exists, file_exists

class PVNet_Config(BasicLoadableObject['PVNet_Config']):
    def __init__(
        self,
        root_dir: str=None,
        blender_path: str=None,
        num_syn: int=100,
        width: int=640, height: int=480,
        low_azi: int=0, high_azi: int=40,
        low_ele: int=-15, high_ele: int=40,
        low_theta: int=10, high_theta: int=40,
        cam_dist: float=0.5,
        min_depth: int=0, max_depth: int=2,
        fx: float=None, fy: float=None,
        cx: float=None, cy: float=None
    ):
        super().__init__()
        # Path Parameters
        if root_dir is None:
            self.config_dir = os.path.dirname(f'{os.path.abspath(__file__)}')
            self.root_dir = os.path.abspath(f'{self.config_dir}/../..')
            self.pkg_dir = f'{self.root_dir}/pvnet_rendering'
        else:
            self.root_dir = root_dir
            self.pkg_dir = f'{self.root_dir}/pvnet_rendering'
            self.config_dir = f'{self.pkg_dir}/config'
    
        self.blender_dir = f'{self.pkg_dir}/blender'
        self.blender_path = blender_path
        self.check_paths()

        # Simulation Parameters
        self.num_syn = num_syn
        self.width, self.height = width, height
        self.low_azi, self.high_azi = low_azi, high_azi
        self.low_ele, self.high_ele = low_ele, high_ele
        self.low_theta, self.high_theta = low_theta, high_theta
        self.cam_dist = cam_dist
        self.min_depth, self.max_depth = min_depth, max_depth

        # Camera Parameters
        self.fx = fx if fx is not None else max(self.width, self.height) / 2
        self.fy = fy if fy is not None else max(self.width, self.height) / 2
        self.cx = cx if cx is not None else self.width / 2
        self.cy = cy if cy is not None else self.height / 2

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ]
        )
    
    @K.setter
    def K(self, K: np.ndarray):
        assert K.shape == (3, 3)
        assert K[0, 1] == 0
        assert K[1, 0] == 0
        assert K[2, 0] == 0
        assert K[2, 1] == 0
        assert K[2, 2] == 1
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def check_paths(self):
        for key, val in self.__dict__.items():
            if key.endswith('_dir'):
                assert dir_exists(val), f"Couldn't find directory at {val}"
            elif key.endswith('_path') and val is not None:
                assert file_exists(val), f"Couldn't find file at {val}"