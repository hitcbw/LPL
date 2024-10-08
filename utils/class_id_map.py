import os
from typing import Dict

__all__ = ["get_class2id_map", "get_id2class_map", "get_n_classes"]


def get_class2id_map(dataset: str, dataset_dir: str = "./dataset") -> Dict[str, int]:
    """
    Args:
        dataset: 
        dataset_dir: the path to the datset directory
    """


    with open(os.path.join(dataset_dir, "{}/mapping.txt".format(dataset)), "r") as f:
        actions = f.read().split("\n")[:-1]

    class2id_map = dict()
    for a in actions:
        class2id_map[a.split()[1]] = int(a.split()[0])

    return class2id_map #经典的读mapping为字典，但是有点捞


def get_id2class_map(dataset: str, dataset_dir: str = "./dataset") -> Dict[int, str]:
    class2id_map = get_class2id_map(dataset, dataset_dir)

    return {val: key for key, val in class2id_map.items()}


def get_n_classes(dataset: str, dataset_dir: str = "./dataset") -> int:
    return len(get_class2id_map(dataset, dataset_dir)) #我靠，只拿长度

def get_classes(dataset: str, dataset_dir: str = "./dataset") -> Dict[str, int]:
    """
    Args:
        dataset:
        dataset_dir: the path to the datset directory
    """
    with open(os.path.join(f"text/detail/{dataset}.txt"), "r") as f:
        actions = f.read().split("\n")
    actions = [a.split(":")[0] for a in actions if a != ""]
    return actions #经典的读mapping为字典，但是有点捞