from __future__ import annotations
from typing import List
from common_utils.base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler

class ObjectConfig(BasicLoadableObject['ObjectConfig']):
    def __init__(self, obj_name: str, ply_path: str, material_path: str=None):
        super().__init__()
        self.obj_name = obj_name
        self.ply_path = ply_path
        self.material_path = material_path

class ObjectConfigList(
    BasicLoadableHandler['ObjectConfigList', 'ObjectConfig'],
    BasicHandler['ObjectConfigList', 'ObjectConfig']
):
    def __init__(self, config_list: List[ObjectConfig]=None):
        super().__init__(obj_type=ObjectConfig, obj_list=config_list)
        self.config_list = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> ObjectConfigList:
        return ObjectConfigList([ObjectConfig.from_dict(item_dict) for item_dict in dict_list])