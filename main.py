import argparse
import os
import random
import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset.class_weight import get_class_weight, get_pos_weight
from dataset.dataset import ActionSegmentationDataset, collate_fn
from libs.helper.LPL import train, validate, test
from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss, KLLoss
from utils.common_tools import import_class, change_label_score
from utils.config import get_config
from utils.getDevice import getDevice, getBase
from utils.optimizer import get_optimizer
from utils.transformer import TempDownSamp, ToTensor

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="train a network for action recognition"
    )
    parser.add_argument("config",  type=str, default="baseline", help="path of a config file")
    parser.add_argument(
        "dataset",
        type=str,
        default='LARA',
        help="a number used to initialize a pseudorandom number generator.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default='Debug',
        help="a number used to initialize a pseudorandom number generator.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="a number used to initialize a pseudorandom number generator.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()

def main() -> None:
    
    args = get_arguments()
    # configuration
    config = get_config(f"config/{args.dataset}/{args.config}.yaml")
    config.device = getDevice()
    base_dir = getBase()
    config.scratch_dir = os.path.join(base_dir, config.scratch_dir)
    config.dataset_dir = os.path.join(base_dir, config.dataset_dir)
    result_path = os.path.join(config.scratch_dir, config.dataset, args.log, 'split' + str(config.split))
    
    print('\n---------------------------result_path---------------------------\n')
    print('result_path:',result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    train_data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(config.ds_rate)]),
        mode="trainval" if not config.param_search else "training",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size if config.device != 'cpu' else 2, #8
        shuffle=True,
        num_workers=config.num_workers, #4
        drop_last=True if config.batch_size > 1 else False,
        collate_fn=collate_fn,
    )

    if config.param_search: #还需要validation
        val_data = ActionSegmentationDataset(
            config.dataset,
            transform=Compose([ToTensor(), TempDownSamp(config.ds_rate)]),
            mode="validation",
            split=config.split,
            dataset_dir=config.dataset_dir,
            csv_dir=config.csv_dir,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
    test_data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(config.ds_rate)]),
        mode="test",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    Model = import_class(config.model)
    CLIP = import_class(config.clip)
    model = Model(config).to(config.device)
    model_text = CLIP(config).to(config.device)
    optimizer = get_optimizer(
        config.optimizer,
        model,
        config.learning_rate,
        momentum=config.momentum,
        dampening=config.dampening,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    ) #Adam或者SGD，这次用的adam
    optimizer2 = get_optimizer(
        'SGD',
        model_text.soa,
        0.01,
        momentum=config.momentum,
        dampening=config.dampening,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    scheduler = CosineAnnealingLR(optimizer2, T_max=50, eta_min=0.001)
    begin_epoch = 0
    best_loss = float("inf") #最好的loss
    if config.dataset == "LARA":
        best_acc = 70.0
        best_F1_50 = 50.0
    elif config.dataset.lower().__contains__("pku"):
        best_acc = 60.0
        best_F1_50 = 50.0
    elif config.dataset.lower().__contains__("tcg"):
        best_acc = 75.0
        best_F1_50 = 65.0
    # Define temporary variables for evaluation scores 最好的三个指标时候的epoch及其参数
    best_test_acc =  {'epoch':0,'train_loss':0,'cls_acc':best_acc,'edit':0,'f1s@0.1':0,'f1s@0.25':0,'f1s@0.5':best_F1_50}
    best_test_F1_50 =  best_test_acc.copy()

    if config.class_weight:
        class_weight = get_class_weight(
            config.dataset,
            split=config.split,
            dataset_dir=config.dataset_dir,
            csv_dir=config.csv_dir,
            mode="training" if config.param_search else "trainval",
        )
        class_weight = class_weight.to(config.device)
    else:
        class_weight = None # 如果没有指定使用类别权重，则将 class_weight 设置为 None

    criterion_cls = ActionSegmentationLoss(
        ce=config.ce,
        focal=config.focal,
        tmse=config.tmse,
        gstmse=config.gstmse,
        weight=class_weight,
        ignore_index=255,
        ce_weight=config.ce_weight,
        focal_weight=config.focal_weight,
        tmse_weight=config.tmse_weight,
        gstmse_weight=config.gstmse,
    ).to(config.device) #包含交叉熵损失以及高斯平滑损失

    pos_weight = get_pos_weight(
        dataset=config.dataset,
        split=config.split,
        csv_dir=config.csv_dir,
        mode="training" if config.param_search else "trainval",
        base_dir=config.dataset_dir
    ).to(config.device) #权重=总数/边界数

    criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight).to(config.device)
    criterion_contrast = KLLoss(reduction='none').to(config.device)
    print("---------- Start training ----------")
    for epoch in range(begin_epoch, config.max_epoch if not args.log=='Debug' else 10):
        start = time.time()
        train_loss, contra_loss = train(
            train_loader,
            model,
            criterion_cls,
            criterion_bound,
            criterion_contrast,
            optimizer,
            opt2=optimizer2,
            config=config,
            model_text=model_text
        ) #读取的函数
        scheduler.step()
        train_time = (time.time() - start) / 60

        if epoch > config.start_eval_epoch:
            start = time.time()
            (
                val_loss,
                cls_acc,
                edit_score,
                segment_f1s,
            ) = validate(
                val_loader,
                model,
                criterion_cls,
                criterion_bound,
                config
            )

            if best_loss > val_loss:
                best_loss = val_loss

            if cls_acc > best_test_acc['cls_acc']:
                print('------------------------------------------------------------------------------------')
                change_label_score(best_test_acc, train_loss, epoch, cls_acc, edit_score, segment_f1s)
                torch.save(
                    model.state_dict(),
                    os.path.join(result_path, 'best_test_acc_model.prm')
                )
                torch.save(
                    model_text.soa.state_dict(),
                    os.path.join(result_path, 'best_test_acc_text.prm')
                )


            if segment_f1s[2] > best_test_F1_50['f1s@0.5']:
                print('------------------------------------------------------------------------------------')
                change_label_score(best_test_F1_50, train_loss, epoch, cls_acc, edit_score, segment_f1s)
                torch.save(
                    model.state_dict(),
                    os.path.join(result_path, 'best_test_F1_0.5_model.prm')
                )
                torch.save(
                    model_text.soa.state_dict(),
                    os.path.join(result_path, 'best_test_F1_0.5_text.prm')
                )

            val_time = (time.time() - start) / 60
            eta_time = (config.max_epoch-epoch)*(train_time+val_time) #剩余时间
            # if you do validation to determine hyperparams
            print(
                'epoch: {}, lr: {:.4f}, train_time: {:.2f}min, train loss: {:.4f}, val_time: {:.2f}min, eta_time: {:.2f}min, \nval_loss: {:.4f}, acc: {:.2f}, edit: {:.2f}, F1@0.1: {:.2f}, F1@0.25: {:.2f}, F1@0.5: {:.2f}, contra_loss: {:.2f}'
                .format(epoch, optimizer.param_groups[0]['lr'], train_time, train_loss, val_time, eta_time, val_loss, cls_acc, \
                edit_score, segment_f1s[0],segment_f1s[1], segment_f1s[2], contra_loss)
            )
            with open(f'{result_path}/scores.txt', "a+") as file:
                file.write('epoch: {}, lr: {:.4f}, train_time: {:.2f}min, train loss: {:.4f}, val_time: {:.2f}min, eta_time: {:.2f}min, \nval_loss: {:.4f}, acc: {:.2f}, edit: {:.2f}, F1@0.1: {:.2f}, F1@0.25: {:.2f}, F1@0.5: {:.2f}, contra_loss: {:.2f}\n'
                .format(epoch, optimizer.param_groups[0]['lr'], train_time, train_loss, val_time, eta_time, val_loss, cls_acc, \
                edit_score, segment_f1s[0],segment_f1s[1], segment_f1s[2], contra_loss)
            )

    print('\n---------------------------best_test_acc---------------------------\n')
    print('{}'.format(best_test_acc))
    print('\n---------------------------best_test_F1_50---------------------------\n')
    print('{}'.format(best_test_F1_50))
    print('\n---------------------------all_train_time---------------------------\n')

    params = torch.load(os.path.join(result_path, 'best_test_acc_model.prm'))
    model.load_state_dict(params)
    test_loss, cls_acc, edit_score, segment_f1s = test(
        test_loader,
        model,
        criterion_cls,
        criterion_bound,
        config
    )
    print(
        'test result on BEST_ACC \n'
        'test_loss: {:.4f}, acc: {:.2f}, edit: {:.2f}, F1@0.1: {:.2f}, F1@0.25: {:.2f}, F1@0.5: {:.2f}'
        .format(test_loss, cls_acc, edit_score, segment_f1s[0], segment_f1s[1], segment_f1s[2])
    )

    params = torch.load(os.path.join(result_path, 'best_test_F1_0.5_model.prm'))
    model.load_state_dict(params)
    test_loss, cls_acc, edit_score, segment_f1s = test(
        test_loader,
        model,
        criterion_cls,
        criterion_bound,
        config
    )
    print(
        'test result on BEST_F1 \n'
        'test_loss: {:.4f}, acc: {:.2f}, edit: {:.2f}, F1@0.1: {:.2f}, F1@0.25: {:.2f}, F1@0.5: {:.2f}'
        .format(test_loss, cls_acc, edit_score, segment_f1s[0], segment_f1s[1], segment_f1s[2])
    )

if __name__ == "__main__":
    main()
