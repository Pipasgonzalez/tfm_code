import argparse
import sys
import itertools

from monai.networks.nets.basic_unet import BasicUNet
# from monai.losses.dice import DiceCELoss, DiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.losses.tversky import TverskyLoss
from monai.losses.dice import DiceFocalLoss, DiceLoss, GeneralizedDiceLoss


from src.utils.data import LMDBTorchDataset, create_balanced_dataloader
from src.networks.unet import get_model
from src.networks.swin_unet import get_swinunet
# from src.networks.ema import MeanTeacher
# from src.utils.losses import AsymmetricUnifiedFocalLoss
from src.helpers import save_batch_images, map_seg_to_color_np, normalize_patch_np, log_batch_to_tensorboard
# from src.utils.losses import DiceLoss

import numpy as np
import random

from torch.utils.data import Subset, Dataset, DataLoader
# import torchvision.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import torchvision.transforms.v2 as v2
import torch.nn.functional as F

# for Mixed Precision Calculation
from torch.amp import autocast, GradScaler
# For Eval
from torchmetrics.segmentation import DiceScore

# For Logging
from torch.utils.tensorboard import SummaryWriter

import shutil
import logging
import os
import csv
from datetime import datetime
from pathlib import Path
import math
import pytz
from sklearn.model_selection import GroupKFold
from skimage.measure import label
import json
from tqdm import tqdm

from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/data/byh_data/SSNet_data/ACDC",
    help="Name of Experiment",
)
parser.add_argument("--exp", type=str, default="BCP", help="experiment_name")

parser.add_argument("--pre_iterations", type=int, default=10000, help="maximum epoch number to train")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="output channel of network")
# label and unlabel
parser.add_argument("--labeled_bs", type=int, default=12, help="labeled_batch_size per gpu")
parser.add_argument("--labelnum", type=int, default=7, help="labeled data")
parser.add_argument("--u_weight", type=float, default=0.5, help="weight of unlabeled pixels")
# costs
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")
parser.add_argument("--magnitude", type=float, default="6.0", help="magnitude")
parser.add_argument("--s_param", type=int, default=6, help="multinum of random masks")


# Custom Args
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--lmdb_path", type=str, default="D:/tfm_data/preprocessed/lmdb_raw")
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--base_lr", type=float, default=0.005, help="segmentation network learning rate")
parser.add_argument("--weight_decay", type=float, default = 0.001)
parser.add_argument("--pretrain_bs", type=int, default=12)
parser.add_argument("--selftrain_bs", type = int, default = 12)
parser.add_argument("--pretrain_epochs", type=int, default=100)
parser.add_argument("--selftrain_epochs", type=int, default = 100)
parser.add_argument("--use_pretrained", action="store_true", help="Whether or not to use the pretrained model for the self training part")
parser.add_argument("--pretrained_model_path", type=str, default = "./src/pretrained_models/exp_20251015_170238/pretrain_best_model.pth")
parser.add_argument("--split", type=int, default=1)
parser.add_argument("--patience", type=int,default=10)
parser.add_argument("--workers", type=int, default = 4)
parser.add_argument('--multiplier_unet', type = int, default=1)
parser.add_argument('--hpc', action='store_true')
parser.add_argument('--output_dir', type=str, default='./src', help='Base directory for outputs (used with --hpc)')
parser.add_argument("--lr_adamw", type = float, default = 0.0008) # As in their paper
parser.add_argument("--lambda_focal", type = float, default = 1)
parser.add_argument("--lambda_dice", type = float, default = 1)
args = parser.parse_args()

COLOR_MAP_NP = np.array([
    [0, 0, 0],       # 0: Background (Black)
    [255, 0, 0],     # 1: Class 1 (Red)
    [0, 255, 0],     # 2: Class 2 (Green)
    [0, 0, 255]      # 3: Class 3 (Blue)
], dtype=np.uint8)

def get_largest_cc(segmentation):
    batch_list = []
    N = segmentation.shape[0]

    # Iterate through batch
    for i in range(0, N):
        class_list = []

        # Iterate through classes, ignoring background though
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(
                temp_seg
            )  # Returns a tensor filled with the scalar value 0, with the same size as input.
            temp_prob[temp_seg == c] = (
                1  # Binary mask of where the prediction was the c-th class
            )
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(
                temp_prob, connectivity=2
            )  # Label connected regions of an integer array.
            if labels.max() != 0:  # Handles the case when the target class actually
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(
                    temp_prob
                )  # Handles the case when the target class does NOT appear

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    
    # Edit because converting from raw list to tensor is apparently very slow
    batch_list_np =np.array(batch_list)

    return torch.Tensor(batch_list_np).long().cuda()


def generate_mask(img):
    batch_size, channel, img_x, img_y = (
        img.shape[0],
        img.shape[1],
        img.shape[2],
        img.shape[3],
    )
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w : w + patch_x, h : h + patch_y] = 0
    loss_mask[:, w : w + patch_x, h : h + patch_y] = 0
    return mask.long(), loss_mask.long()

def bcp_loss(
    output,
    img_l,
    patch_l,
    mask,
    ce_loss,  # Must be initialized with reduction='none'
    dc_loss,  # Must be a DiceLoss instance
    l_weight=1.0,
    u_weight=0.5, # CHANGE : This was 0.5 before, but since we have so much more unlabeled data, maybe a lower weight is better
    unlab=False,
):
    # Determine weights for the two regions
    image_weight, patch_weight = (l_weight, u_weight) if not unlab else (u_weight, l_weight)
    patch_mask = 1 - mask

    loss_ce = image_weight * (ce_loss(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (ce_loss(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)

    output_soft = F.softmax(output, dim=1)
    img_l_onehot = F.one_hot(img_l, num_classes=output.shape[1]).permute(0, 3, 1, 2)
    patch_l_onehot = F.one_hot(patch_l, num_classes=output.shape[1]).permute(0, 3, 1, 2)
    loss_dice_img = dc_loss(
        output_soft * mask.unsqueeze(1),
        img_l_onehot * mask.unsqueeze(1)
    )
    loss_dice_patch = dc_loss(
        output_soft * patch_mask.unsqueeze(1),
        patch_l_onehot * patch_mask.unsqueeze(1)
    )
    total_loss_dice = image_weight * loss_dice_img + patch_weight * loss_dice_patch

    return total_loss_dice, loss_ce

def pre_train(args, labeled_wsi_ids, test_wsi_ids, logs_dir,model_save_dir,skip_validation=False, continue_training = False, pretrain_path:str = "./src/pretrained_models/20251025_183955_split2_modelswinunet_15epochs_6bs/pretrain_best_model.pth"):
    if args.pretrain_bs % 3 != 0:
        raise RuntimeError("Pretrain Batch Size needs to be dividable by 3")
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    num_classes = args.num_classes
    lmdb_path = args.lmdb_path
    patience = args.patience
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Batch Size
    bs = args.pretrain_bs
    optimizer = args.optimizer
    workers = args.workers
    multiplier = args.multiplier_unet
    model_type = args.model
    max_epoch = args.pretrain_epochs
    focal_weight = args.lambda_focal # Focal Loss ca 1/4 of the dice loss CHANGE by having higher batch size it becomes clear that focal is only about 2x of dice (at batch size 18)
    dice_weight = args.lambda_dice

    if not args.hpc:
        # Multi Resolution Setup
        train_dataset_3 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl6", allowed_wsi_ids=labeled_wsi_ids, ds_type='train')
        train_dataset_4 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl4_fmt", allowed_wsi_ids=labeled_wsi_ids, ds_type='train')
        train_dataset_5 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl5_fmt", allowed_wsi_ids=labeled_wsi_ids, ds_type='train')
        train_dl_4 = create_balanced_dataloader(dataset=train_dataset_4, batch_size=int(bs/3), replacement=True, num_workers=workers)
        train_dl_3 = create_balanced_dataloader(dataset=train_dataset_3, batch_size=int(bs/3), replacement=True, num_workers=workers,total_images=len(train_dataset_4))
        train_dl_5 = create_balanced_dataloader(dataset=train_dataset_5, batch_size=int(bs/3), replacement=True, num_workers=workers,total_images=len(train_dataset_4))
        
        test_dataset_3 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl6", allowed_wsi_ids=test_wsi_ids, ds_type='test')
        test_loader_3 = DataLoader(dataset=test_dataset_3, batch_size=bs, num_workers=2, pin_memory=True, persistent_workers=True)
        test_dataset_4 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl4_fmt", allowed_wsi_ids=test_wsi_ids, ds_type='test')
        test_loader_4 = DataLoader(dataset=test_dataset_4, batch_size=bs, num_workers=2, pin_memory=True, persistent_workers=True)
        test_dataset_5 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl5_fmt", allowed_wsi_ids=test_wsi_ids, ds_type='test')
        test_loader_5 = DataLoader(dataset=test_dataset_5, batch_size=bs, num_workers=2, pin_memory=True, persistent_workers=True)
    else:
        # Multi Resolution Setup
        train_dataset_3 = LMDBTorchDataset(lmdb_path="./dataset_lvl3_fmt", allowed_wsi_ids=labeled_wsi_ids, ds_type='train')
        train_dataset_4 = LMDBTorchDataset(lmdb_path="./dataset_lvl4_fmt", allowed_wsi_ids=labeled_wsi_ids, ds_type='train')
        train_dataset_5 = LMDBTorchDataset(lmdb_path="./dataset_lvl5_fmt", allowed_wsi_ids=labeled_wsi_ids, ds_type='train')
        train_dl_4 = create_balanced_dataloader(dataset=train_dataset_4, batch_size=int(bs/3), replacement=True, num_workers=workers)
        train_dl_3 = create_balanced_dataloader(dataset=train_dataset_3, batch_size=int(bs/3), replacement=True, num_workers=workers, total_images=len(train_dataset_4))
        train_dl_5 = create_balanced_dataloader(dataset=train_dataset_5, batch_size=int(bs/3), replacement=True, num_workers=workers, total_images=len(train_dataset_4))

        test_dataset_3 = LMDBTorchDataset(lmdb_path="./dataset_lvl3_fmt", allowed_wsi_ids=test_wsi_ids, ds_type='test')
        test_dataset_4 = LMDBTorchDataset(lmdb_path="./dataset_lvl4_fmt", allowed_wsi_ids=test_wsi_ids, ds_type='test')
        test_dataset_5 = LMDBTorchDataset(lmdb_path="./dataset_lvl5_fmt", allowed_wsi_ids=test_wsi_ids, ds_type='test')
        test_loader_3 = DataLoader(dataset=test_dataset_3, batch_size=bs, num_workers=2, pin_memory=True)
        test_loader_4 = DataLoader(dataset=test_dataset_4, batch_size=bs, num_workers=2, pin_memory=True)
        test_loader_5 = DataLoader(dataset=test_dataset_5, batch_size=bs, num_workers=2, pin_memory=True)

    logger.info(f"Length of Resolution 3 Train Dataset {len(train_dataset_3)}")
    logger.info(f"Length of Resolution 4 Train Dataset {len(train_dataset_4)}")
    logger.info(f"Length of Resolution 5 Train Dataset {len(train_dataset_5)}")
    if not continue_training:
        if model_type == 'unet':
            model = get_model(nclasses=num_classes, device="cuda", multiplier = multiplier)
        elif model_type == 'swinunet':
            if args.hpc:
                model = get_swinunet()
            else:
                model = get_swinunet()
        else:
            raise ValueError(f'No implementation currently for Network Type {model}')
    else:
        if model_type == 'unet':
            model = get_model(nclasses=num_classes, device="cuda", multiplier = multiplier)
            model.load_state_dict(torch.load(pretrain_path))
        elif model_type == 'swinunet':
            if args.hpc:
                model = get_swinunet()
                model.load_state_dict(torch.load(pretrain_path))
            else:
                model = get_swinunet()
                model.load_state_dict(torch.load(pretrain_path))
        else:
            raise ValueError(f'No implementation currently for Network Type {model}')


    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) # Directly from the original paper of swin unet and BCP``
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),lr=args.lr_adamw,weight_decay=1e-4 # momentum=0.9, weight_decay=0.0001
        )
    else:
        raise ValueError("Optimizer has to be either sgd or adamw")

    # ce_weights = torch.from_numpy(np.array([2, 0.1, 0.1])) # from class_distributions.ipynb
    
    focal_loss_fn = FocalLoss(
        include_background=True, 
        to_onehot_y=True, 
        use_softmax=True, 
    )
    dice_loss_fn = GeneralizedDiceLoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        batch=True,
    )

    dice_eval = DiceScore(num_classes=4, include_background=True, input_format="index", average="none") # This version aggregated dice scores per class to log them out properly
    # Scaler
    scaler = GradScaler(device="cuda")  # manages scaling automatically

    # Set Up Writer for TensorBoard Monitoring
    tensorboard_path = os.path.join(logs_dir, "pretrain")
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)

    # Model Output Path
    model_path = os.path.join(model_save_dir, "pretrain_best_model.pth")

    iter_num = 0
    eval_num = 0  # max_iterations // len(pseudo_unlabeled_ds) + 1
    best_performance = 0.0
    patience_counter = 0

    for i in range(max_epoch):
        epoch_losses = []
        model.cuda()
        model.train()

        # progress_bar_train = tqdm(train_loader,desc=f"Epoch {i+1}/{max_epoch}")
        progress_bar_train = tqdm(zip(train_dl_3, train_dl_4, train_dl_5), desc=f"Epoch {i+1}/{max_epoch}", total=max(len(train_dl_3), len(train_dl_4)))
        for (patch_3,target_3),(patch_4,target_4),(patch_5, target_5) in progress_bar_train:
            if torch.isnan(patch_3).any() or torch.isnan(patch_4).any() or torch.isnan(patch_5).any():
                logger.error(f"NaN in INPUT patch at epoch {i}, skipping batch")
                continue

            optimizer.zero_grad()
            patch_3,target_3, patch_4, target_4, patch_5, target_5 =patch_3.cuda(), target_3.long().cuda(), patch_4.cuda(), target_4.long().cuda(), patch_5.cuda(), target_5.long().cuda()

            # Stack them together
            patch = torch.cat((patch_3, patch_4, patch_5), dim=0)
            target = torch.cat((target_3, target_4, target_5), dim=0)

            # Add this line to log GPU memory usage at the start of the first epoch
            if i == 0 and iter_num == 5:
                max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                logger.info(f"Initial Memory Usage (Batch Size {args.pretrain_bs}): {max_memory:.2f} MiB")

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                out= model(patch)
                dice_loss = dice_loss_fn(out, target.unsqueeze(1)) * dice_weight
                focal_loss = focal_loss_fn(out,target.unsqueeze(1)) * focal_weight
                total_loss = dice_loss + focal_loss
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN/Inf loss at epoch {i}, batch {iter_num}, skipping")
                continue
            # dab_loss_value.backward()
            # optimizer.step()
            # # Training Loop
            scaler.scale(total_loss).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()

            iter_num += 1

            writer.add_scalar("Loss/Pretrain/Total", total_loss.item(), iter_num)
            writer.add_scalars("Loss/Pretrain/FocalVsDice", {"Dice":dice_loss.item(), "Focal":focal_loss.item()},iter_num)
            progress_bar_train.set_postfix({"Batch Loss": f"{total_loss.item():.4f}"})
            epoch_losses.append(total_loss.item())
        # Begin Eval
        model.eval()
        writer.add_scalars("Loss/Total", {'Train':np.mean(epoch_losses)}, i+1)

        if not skip_validation:
            patches_to_log_per_loader = 4  # Log 4 random images per loader (total up to 12)
            val_losses_per_resolution = []
            loaders = [
                (test_loader_3, "res3"),
                (test_loader_4, "res4"),
                (test_loader_5, "res5")
            ]

            with torch.no_grad():
                dice_eval.reset()
                val_loss_values = []

                for loader, res_tag in loaders:
                    num_batches = len(loader)
                    if num_batches < patches_to_log_per_loader:
                        log_indices_set = set(range(num_batches))
                    else:
                        log_indices = np.random.choice(num_batches, patches_to_log_per_loader, replace=False)
                        log_indices_set = set(log_indices)

                    progress_bar_val = tqdm(loader, 
                                            desc=f"Running Validation on Epoch: {i+1} ({res_tag})", 
                                            total=num_batches)

                    for val_idx, batch in enumerate(progress_bar_val):
                        eval_num += 1

                        images, targets, wsi_ids = batch
                        images, targets = images.cuda(), targets.cuda()

                        with autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = model(images)
                            dice_loss_eval = dice_loss_fn(out, targets.unsqueeze(1)) * dice_weight
                            focal_loss_eval = focal_loss_fn(out, targets.unsqueeze(1)) * focal_weight
                            total_loss_eval = focal_loss_eval + dice_loss_eval
                            pred_prob_eval = F.softmax(out, dim=1)
                            pred = torch.argmax(pred_prob_eval, dim=1)

                        val_loss_values.append(total_loss_eval.item())
                        dice_eval.update(pred, targets)

                        if val_idx in log_indices_set:
                            targets_np = targets.cpu().numpy()
                            pred_np = pred.cpu().numpy()
                            images_np = images.cpu().numpy()

                            b = 0 

                            target_patch_np = targets_np[b]
                            image_patch_np = images_np[b]
                            pred_patch_np = pred_np[b]

                            # 1. Normalize original image (NumPy)
                            img_normalized = normalize_patch_np(image_patch_np)

                            # 2. Color-map ground truth (NumPy)
                            target_color = map_seg_to_color_np(target_patch_np, COLOR_MAP_NP)

                            # 3. Color-map prediction (NumPy)
                            pred_color = map_seg_to_color_np(pred_patch_np, COLOR_MAP_NP)

                            # 4. Log to TensorBoard
                            tag_prefix = f"Validation/{res_tag}_Batch{val_idx}"
                            writer.add_image(f"{tag_prefix}_Image", img_normalized, i + 1, dataformats='CHW')
                            writer.add_image(f"{tag_prefix}_GroundTruth", target_color, i + 1, dataformats='CHW')
                            writer.add_image(f"{tag_prefix}_Prediction", pred_color, i + 1, dataformats='CHW')

                    per_class_dice = dice_eval.compute()
                    mean_dice = torch.mean(per_class_dice[1:]).item()
                    dice_dict = {
                        'Macro': mean_dice,
                        'Class0': per_class_dice[0].item(),
                        'Class1': per_class_dice[1].item(),
                        'Class2': per_class_dice[2].item(),
                        'Class3': per_class_dice[3].item()
                    }
                    writer.add_scalars(f"Eval/DiceScoreEval_{res_tag}", dice_dict, i + 1)
                    val_losses_per_resolution.append(np.mean(val_loss_values))

                    # Gets added on top of the Training Loss
                writer.add_scalars("Loss/Total", {'Val': np.mean(val_losses_per_resolution)}, i + 1)

            if mean_dice > best_performance:
                best_performance = mean_dice
                patience_counter = 0
                logger.info(f"Saving New Best Pre-Train Model with Dice Score of {mean_dice}")
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter +=1
            
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
        # Just to be sure
        model.train()
    
    # Return Saved Model Path for subsequent self training
    return model_path


if __name__ == "__main__":
    SPLIT = args.split
    # FOR TRAINING MERGED MULTI RESOLUTION DATASET:
    # python -m src.training_scripts.Baseline_MultiRes --model swinunet --pretrain_bs 6 --workers 0 --optimizer adamw --pretrain_epochs 50 --split 3
    if args.hpc:
        output_base = args.output_dir
    else:
        output_base = "./src"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CHANGE sortable experiment name
    experiment_name = f"{timestamp}_split{SPLIT}_model{args.model}_{args.pretrain_epochs}epochs_{args.pretrain_bs}bs"

    experiment_Logs_dir = os.path.join(output_base, "logs", experiment_name)
    clean_path = experiment_Logs_dir.replace("\\", "/")
    model_save_dir = os.path.join(output_base, "pretrained_models", experiment_name)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(experiment_Logs_dir, exist_ok=True)
    logger.info(f"Saving Model Logs to: {clean_path}")    
    # Log Parameters to the CSV File
    if not args.hpc:
        import csv
        data= {
            "experiment_name": experiment_name,
            "datetime": timestamp,
            "model": args.model,
            "epochs": args.pretrain_epochs,
            "batchsize": args.pretrain_bs,
            "optimizer": args.optimizer,
            "learning_rate": args.lr_adamw if args.optimizer == "adamw" else args.lr_sgd,
            "split": args.split 
        }
        file_path = "./src/experiments_logging/experiments.csv"
        is_file_empty = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        with open(file_path, "a", newline="") as csvfile:
            fieldnames= ["experiment_name", "datetime", "model", "epochs", "batchsize", "optimizer", "learning_rate", "split"]
            writer = csv.DictWriter(csvfile,fieldnames)
            if is_file_empty:
                writer.writeheader()
            writer.writerow(data)
            print("Data Logger")
    else:
        # HPC branch: create a new CSV in HPC-specific folder
        import csv
        data = {
            "experiment_name": experiment_name,
            "datetime": timestamp,
            "model": args.model,
            "epochs": args.pretrain_epochs,
            "batchsize": args.pretrain_bs,
            "optimizer": args.optimizer,
            "learning_rate": args.lr_adamw if args.optimizer == "adamw" else args.lr_sgd,
            "split": args.split 
        }
        file_path = os.path.join(output_base, "experiment_setup.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", newline="") as csvfile:
            fieldnames = ["experiment_name", "datetime", "model", "epochs", "batchsize", "optimizer", "learning_rate", "split"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
            print(f"New HPC CSV created: {file_path}")



    # Load the split and stuff
    with open("./src/config/splits_3110.json") as f:
        splits = json.load(f)

    selected_splits = splits.get(f"split_{SPLIT}")
    if selected_splits is None:
        raise ValueError(f"Could not find split {SPLIT} in config file")
    labeled_wsi = selected_splits["labeled_wsi_ids"]
    test_wsi = selected_splits["test_wsi_ids"]
    # unlabeled_wsi = selected_splits["unlabeled_wsi_ids"]

    if args.deterministic:
        cudnn.benchmark = True # Can be changed to false
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not args.use_pretrained:
        pretrained_model_path = pre_train(
            args, labeled_wsi_ids=labeled_wsi,test_wsi_ids = test_wsi, logs_dir=experiment_Logs_dir,model_save_dir=model_save_dir, skip_validation=False
        )
