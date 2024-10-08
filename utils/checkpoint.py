import os
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(save_states, os.path.join(result_path, "checkpoint.pth"))


def resume(
    result_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[Any]:

    resume_path = os.path.join(result_path, "checkpoint.pth") # 构建 checkpoint 文件路径
    print("loading checkpoint {}".format(resume_path)) # 打印正在加载的 checkpoint 文件路径
    # 使用 torch.load 加载 checkpoint 文件，map_location 用于指定加载的设备
    checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
    # 从 checkpoint 中获取开始训练的 epoch、最佳损失、模型的状态字典
    begin_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])

    # confirm whether the optimizer matches that of checkpoints
    optimizer.load_state_dict(checkpoint["optimizer"]) # 确保 optimizer 与 checkpoint 中保存的一致

    return begin_epoch, model, optimizer, best_loss # 返回从 checkpoint 中获取的信息：开始的 epoch、模型、优化器、最佳损失
