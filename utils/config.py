import dataclasses
import pprint
from typing import Any, Dict, Tuple

import yaml

__all__ = ["get_config"]


@dataclasses.dataclass
class Config:
    model: str = "ActionSegmentRefinementNetwork"
    n_layers: int = 10
    backbone: str = "none"
    backbone_args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    n_refine_layers: int = 10
    n_stages: int = 4
    n_features: int = 64
    n_stages_asb: int = 4
    n_stages_brb: int = 4
    none_idx: int = 9999
    learnable_n_ctx: int = 10
    SFI_layer: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    scratch_dir: str = '/'
    # loss function
    ce: bool = True  # cross entropy
    ce_weight: float = 1.0
    hard_classes: str = ''
    focal: bool = False
    focal_weight: float = 1.0
    n_classes: int = 0
    contra_start_epoch: int = 0
    tmse: bool = False  # temporal mse
    tmse_weight: float = 0.15
    n_contra_features: int = 512
    clip_norm: bool = False
    gstmse: bool = True  # gaussian similarity loss
    gstmse_weight: float = 1.0
    gstmse_index: str = "feature"  # similarity index
    start_eval_epoch: int = 0
    # if you use class weight to calculate cross entropy or not
    class_weight: bool = True
    seg_que_size: int = 1000
    batch_size: int = 1
    prompt_num: int = 1
    with_loss_weight: bool = False
    with_optim2: bool = False
    # the number of input feature channels
    in_channel: int = 2048
    prompt_init: str = "random"
    num_workers: int = 0
    max_epoch: int = 50

    optimizer: str = "Adam"
    optimizer2: str = "SGD"
    balanceCE: bool = True
    learning_rate: float = 0.0005
    momentum: float = 0.9  # momentum of SGD
    dampening: float = 0.0  # dampening for momentum of SGD
    weight_decay: float = 0.0001  # weight decay
    nesterov: bool = True  # enables Nesterov momentum
    clip: str = 'OriginalClip'
    param_search: bool = False

    # thresholds for calcualting F1 Score
    iou_thresholds: Tuple[float, ...] = (0.1, 0.25, 0.5)
    anchor_cls: bool = False
    anchor_text: bool = False
    anchor_contra: bool = False
    # boundary regression
    tolerance: int = 5
    boundary_th: float = 0.5
    lambda_b: float = 0.1

    dataset: str = "MCFS-22"
    dataset_dir: str = "./dataset"
    csv_dir: str = "./csv"
    split: int = 1
    ds_rate: int = 1
    result_path: str = "./config"
    device: int = 0
    refinement_method: str = "refinement_with_boundary"
    temperature: float = 0.1
    with_seg_cls: bool = False
    with_contra: bool = False
    part_list: Tuple[list, ...] = ([1],[2])
    part_text_idx: Tuple[int, ...] = (0, 1, 2, 3, 4)
    n_node: int = -1
    head_list: Tuple[int, ...] = (9, 10)
    hand_list: Tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17, 18)
    foot_list: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    hip_list: Tuple[int, ...] = (0, 9, 1, 5)

    def __post_init__(self) -> None:
        self._type_check()

        print("-" * 10, "Experiment Configuration", "-" * 10)
        pprint.pprint(dataclasses.asdict(self), width=1)

    def _type_check(self) -> None:
        """Reference:
        https://qiita.com/obithree/items/1c2b43ca94e4fbc3aa8d
        """

        _dict = dataclasses.asdict(self)

        for field, field_type in self.__annotations__.items():
            # if you use type annotation class provided by `typing`,
            # you should convert it to the type class used in python.
            # e.g.) Tuple[int] -> tuple
            # https://stackoverflow.com/questions/51171908/extracting-data-from-typing-types

            # check the instance is Tuple or not.
            # https://github.com/zalando/connexion/issues/739
            if hasattr(field_type, "__origin__"):
                # e.g.) Tuple[int].__args__[0] -> `int`
                element_type = field_type.__args__[0]

                # e.g.) Tuple[int].__origin__ -> `tuple`
                field_type = field_type.__origin__

                self._type_check_element(field, _dict[field], element_type)

            # bool is the subclass of int,
            # so need to use `type() is` instead of `isinstance`
            if type(_dict[field]) is not field_type:
                raise TypeError(
                    f"The type of '{field}' field is supposed to be {field_type}."
                )

    def _type_check_element(
            self, field: str, vals: Tuple[Any], element_type: type
    ) -> None:
        for val in vals:
            if type(val) is not element_type:
                raise TypeError(
                    f"The element of '{field}' field is supposed to be {element_type}."
                )


# 函数遍历输入的 _dict 中的每一对键值对，检查值（val）是否是列表。如果是列表，就将该值转换为元组。最后，返回修改后的 _dict。
# 这种转换的目的可能是为了确保某些数据结构在使用时是不可变的，以避免一些不可预测的问题，尤其是在使用类似 dataclasses 这样的工具时。
def convert_list2tuple(_dict: Dict[str, Any]) -> Dict[str, Any]:
    # cannot use list in dataclass because mutable defaults are not allowed.
    for key, val in _dict.items():
        if isinstance(val, list):
            _dict[key] = tuple(val)

    return _dict


def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)  # 从 YAML 文件中加载配置信息成为字典

    config_dict = convert_list2tuple(config_dict)  # 将字典中的所有列表转换成元组
    config = Config(**config_dict)  # 使用 Config 类创建一个配置对象，并将字典中的内容传递给 Config 的构造函数#
    return config
