import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float = 0.9,
    dampening: float = 0.0,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
) -> optim.Optimizer:

    assert optimizer_name in ["SGD", "Adam"]
    print(f"{optimizer_name} will be used as an optimizer.")

    if optimizer_name == "Adam":
        optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum, #通过引入一个动量项，使得模型在更新时会保留一部分上一次更新的方向
            dampening=dampening, #引入了一个摩擦项，目的是在优化过程中防止振荡或过冲
            weight_decay=weight_decay, #对模型的权重添加一个惩罚项，降低模型的复杂度
            nesterov=nesterov, #对标准动量的一种改进，主要在计算梯度时引入一个校正项，以更准确地估计梯度
        )

    return optimizer

def get_optimizer_prompt(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float = 0.9,
    dampening: float = 0.0,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
) -> optim.Optimizer:
    optimizer = optim.SGD(
        model.extra,
        lr=learning_rate,
        momentum=momentum,  # 通过引入一个动量项，使得模型在更新时会保留一部分上一次更新的方向
        dampening=dampening,  # 引入了一个摩擦项，目的是在优化过程中防止振荡或过冲
        weight_decay=weight_decay,  # 对模型的权重添加一个惩罚项，降低模型的复杂度
        nesterov=nesterov,  # 对标准动量的一种改进，主要在计算梯度时引入一个校正项，以更准确地估计梯度
    )


    return optimizer