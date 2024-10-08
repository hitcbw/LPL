import torch
def get_node_dict(dataset):
    if dataset.lower().__contains__('pku'):
        node = 25
        head_list = torch.Tensor([2, 3, 20]).long()
        hand_list = torch.Tensor([4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24]).long()
        foot_list = torch.Tensor([12, 13, 14, 15, 16, 17, 18, 19]).long()
        hip_list = torch.Tensor([0, 1, 20, 12, 16]).long()
        global_list = torch.arange(node).long()
        node_num_list = [3, 12, 8, 5, 25]
        node_idx_list = [head_list, hand_list, foot_list, hip_list, global_list]
    elif dataset.lower().__contains__('lara'):
        node = 19
        head_list = torch.Tensor([9, 10]).long()
        hand_list = torch.Tensor([11, 12, 13, 14, 15, 16, 17, 18]).long()
        foot_list = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).long()
        hip_list = torch.Tensor([0, 9, 1, 5]).long()
        global_list = torch.arange(node).long()
        node_num_list = [2, 8, 8, 4, 19]
        node_idx_list = [head_list, hand_list, foot_list, hip_list, global_list]
    elif dataset.lower().__contains__('tcg'):
        node = 17
        head_list = torch.Tensor([0, 10, 12, 13, ]).long()  # lhand
        hand_list = torch.Tensor([0, 3, 5, 6]).long()  # rhand
        foot_list = torch.Tensor([0, 2, 8, 15, 16]).long()
        hip_list = torch.Tensor([2, 4, 7, 8, 11, 14, 15]).long()
        global_list = torch.arange(node).long()
        node_num_list = [4, 4, 5, 7, 17]
        node_idx_list = [head_list, hand_list, foot_list, hip_list, global_list]
    else:
        node = 25
        head_list = torch.Tensor([0, 15, 16, 17, 18]).long()
        hand_list = torch.Tensor([1, 2, 3, 4, 5, 6, 7]).long()
        foot_list = torch.Tensor([9, 10, 11, 22, 23, 24, 8, 12, 13, 14, 19, 20, 21]).long()
        hip_list = torch.Tensor([0, 1, 8, 9, 12]).long()
        global_list = torch.arange(node).long()
        node_num_list = [3, 12, 8, 5, 25]
        node_idx_list = [head_list, hand_list, foot_list, hip_list, global_list]
    return {
        'node': node,
        'node_num_list': node_num_list,
        'node_idx_list': node_idx_list
    }