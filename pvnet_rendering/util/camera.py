from __future__ import annotations
import numpy as np
from common_utils.base.basic import BasicLoadableObject
from common_utils.file_utils import file_exists
from typing import List

class BasicCamera(BasicLoadableObject['BasicCamera']):
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def to_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ]
        )
    
    @classmethod
    def from_matrix(self, arr: np.ndarray) -> BasicCamera:
        assert arr.shape == (3, 3)
        assert arr[0, 1] == 0
        assert arr[1, 0] == 0
        assert arr[2, 0] == 0
        assert arr[2, 1] == 0
        assert arr[2, 2] == 1
        return BasicCamera(
            fx=arr[0,0],
            fy=arr[1, 1],
            cx=arr[0, 2],
            cy=arr[1, 2]
        )
    
    @classmethod
    def from_image_shape(self, image_shape: List[int]) -> BasicCamera:
        height, width = image_shape[:2]
        return BasicCamera(
            fx=max([width, height])/2,
            fy=max([width, height])/2,
            cx=width/2,
            cy=height/2
        )
    
    def save_to_txt(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            raise FileExistsError(
                f"""
                File already exists at {save_path}
                Hint: Use overwrite=True to save anyway.
                """
            )
        np.savetxt(fname=save_path, X=self.to_matrix())

    @classmethod
    def load_from_txt(self, load_path: str) -> BasicCamera:
        if not file_exists(load_path):
            raise FileNotFoundError(f"Couldn't find file at {load_path}")
        mat = np.loadtxt(load_path)
        return BasicCamera.from_matrix(mat)