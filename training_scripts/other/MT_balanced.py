import argparse
import sys

from monai.networks.nets.basic_unet import BasicUNet
# from monai.losses.dice import DiceCELoss, DiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.losses.tversky import TverskyLoss
from monai.losses.dice import DiceFocalLoss


from src.utils.data import LMDBTorchDataset, create_balanced_dataloader
from src.networks.unet import get_model
from src.networks.swin_unet import get_swinunet, get_swinunet_max_beef
# from src.networks.ema import MeanTeacher
# from src.utils.losses import AsymmetricUnifiedFocalLoss
from src.helpers import save_batch_images, map_seg_to_color_np, normalize_patch_np
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
parser.add_argument("--weight_decay", type=float, default = 1e-4)
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

def pre_train(args, labeled_wsi_ids, test_wsi_ids, logs_dir,model_save_dir,skip_validation=False):
    if args.pretrain_bs % 2 != 0:
        raise RuntimeError("Pretrain Batch Size needs to be dividable by 2")
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

    train_dataset = LMDBTorchDataset(lmdb_path=lmdb_path, allowed_wsi_ids=labeled_wsi_ids, ds_type='train')
    train_dl = create_balanced_dataloader(dataset=train_dataset, batch_size=bs, replacement=True, num_workers=workers)
    # train_loader = create_balanced_dataloader(train_dataset, batch_size=bs, num_workers=workers, replacement=True)
    test_dataset = LMDBTorchDataset(lmdb_path=lmdb_path, allowed_wsi_ids=test_wsi_ids, ds_type='test')

    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=workers, pin_memory=True)

    if model_type == 'unet':
        model = get_model(nclasses=num_classes, device="cuda", multiplier = multiplier)
    elif model_type == 'swinunet':
        model = get_swinunet_max_beef()
    else:
        raise ValueError(f'No implementation currently for Network Type {model}')

    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) # Directly from the original paper
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),# lr=base_lr,weight_decay=weight_decay # momentum=0.9, weight_decay=0.0001
        )
    else:
        raise ValueError("Optimizer has to be either sgd or adamw")


    # ce_weights = torch.from_numpy(np.array([2, 0.1, 0.1])) # from class_distributions.ipynb
    # unified_loss = AsymmetricUnifiedFocalLoss(lambda_weight=0.5,delta=0.6,gamma=0.5,num_classes=4,rare_class_idx=1,background_class_idx=0)

    # foc_loss = FocalLoss(include_background=False,to_onehot_y=True,use_softmax=True,weight=ce_weights) # CHANGE removed weights
    tvsky_loss = TverskyLoss(include_background=False,to_onehot_y=True,softmax=True)

    # New DiceFocal
    dab_loss = DiceFocalLoss(include_background=False,to_onehot_y=True,softmax=True)

    # dice_eval = DiceScore(num_classes=4, include_background=False, input_format="index", average="macro")
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
    eval_num = 0
    max_epoch = args.pretrain_epochs  # max_iterations // len(pseudo_unlabeled_ds) + 1
    best_performance = 0.0
    patience_counter = 0

    for i in range(max_epoch):
        epoch_losses = []
        model.cuda()
        model.train()

        # progress_bar_train = tqdm(train_loader,desc=f"Epoch {i+1}/{max_epoch}")
        progress_bar_train = tqdm(train_dl, desc=f"Epoch {i+1}/{max_epoch}")
        for (patch,target) in progress_bar_train:

            optimizer.zero_grad()
            patch,target=patch.cuda(), target.cuda()

            with autocast(device_type="cuda", dtype=torch.float16):
                out= model(patch)
                dab_loss_value = dab_loss(out,target.unsqueeze(1))

            # dab_loss_value.backward()
            # optimizer.step()
            # # Training Loop
            scaler.scale(dab_loss_value).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer=optimizer)
            scaler.update()

            iter_num += 1

            writer.add_scalar("Loss/Pretrain/Total", dab_loss_value.item(), iter_num)
            progress_bar_train.set_postfix({"Batch Loss": f"{dab_loss_value.item():.4f}"})
            epoch_losses.append(dab_loss_value.item())
        # Begin Eval
        model.eval()
        writer.add_scalars("Loss/Total", {'Train':np.mean(epoch_losses)}, i+1)

        # To avoid gradient calculation etc.
        if not skip_validation:
            images_logged_this_epoch = 0
            patches_to_log = 5  # How many patches to log
            with torch.no_grad():
                dice_eval.reset()
                val_loss_values = []
                dice_scores = []  # store all batch Dice scores
                debug_info_printed = False
                progress_bar_val = tqdm(test_loader, desc=f"Running Validation on Epoch: {i+1}")
                last_log_batch_idx = -13
                for val_idx,batch in enumerate(progress_bar_val):
                    eval_num += 1
                    images, targets = batch
                    images, targets = images.cuda(), targets.cuda()
                    with autocast(device_type="cuda",dtype=torch.float16):
                        out = model(images)
                        val_loss = dab_loss(out, targets.unsqueeze(1))
                        pred_prob = F.softmax(out, dim=1)
                        pred = torch.argmax(pred_prob, dim=1)

                    # dice_score = dice_eval(pred, targets) DEPRECATED
                    val_loss_values.append(val_loss.item())
                    dice_eval.update(pred, targets)
                    per_class_dice = dice_eval.compute()
                    mean_dice = torch.mean(per_class_dice[1:]).item()


                    # Image Showing
                    targets_np = targets.cpu().numpy()
                    pred_np = pred.cpu().numpy()
                    images_np = images.cpu().numpy()

                    if images_logged_this_epoch < patches_to_log and (val_idx - last_log_batch_idx) > 10:
                        # Iterate through batch (now NumPy arrays)
                        for b in range(images_np.shape[0]):
                            if images_logged_this_epoch >= patches_to_log:
                                break 

                            target_patch_np = targets_np[b]  # (D, H, W) or (H, W)
                            
                            class_1_pixels = np.sum(target_patch_np == 1)
                            total_pixels = target_patch_np.size # .numel() equivalent
                            
                            if total_pixels > 0 and (class_1_pixels / total_pixels) >= 0.1:
                                # Found a good patch
                                print(f"\n[Epoch {i+1}]: Found patch with {(class_1_pixels / total_pixels)*100:.2f}% class 1. Logging image {images_logged_this_epoch}...")
                                
                                image_patch_np = images_np[b]  # (C, D, H, W) or (C, H, W)
                                pred_patch_np = pred_np[b]    # (D, H, W) or (H, W)

                                # Handle 3D or 2D. If 3D, take middle slice.
                                is_3d = image_patch_np.ndim == 4
                                if is_3d:
                                    mid_slice_idx = image_patch_np.shape[1] // 2
                                    image_to_log_np = image_patch_np[:, mid_slice_idx, :, :]
                                    target_to_log_np = target_patch_np[mid_slice_idx, :, :]
                                    pred_to_log_np = pred_patch_np[mid_slice_idx, :, :]
                                else:
                                    image_to_log_np = image_patch_np
                                    target_to_log_np = target_patch_np
                                    pred_to_log_np = pred_patch_np

                                # 1. Normalize original image (NumPy)
                                img_normalized = normalize_patch_np(image_to_log_np)
                                
                                # 2. Color-map ground truth (NumPy)
                                target_color = map_seg_to_color_np(target_to_log_np, COLOR_MAP_NP)
                                
                                # 3. Color-map prediction (NumPy)
                                pred_color = map_seg_to_color_np(pred_to_log_np, COLOR_MAP_NP)

                                # 4. Log to TensorBoard (writer accepts NumPy arrays)
                                tag_prefix = f"Validation/Patch_{images_logged_this_epoch}"
                                writer.add_image(f"{tag_prefix}_Image", img_normalized, i + 1, dataformats='CHW')
                                writer.add_image(f"{tag_prefix}_GroundTruth", target_color, i + 1, dataformats='CHW')
                                writer.add_image(f"{tag_prefix}_Prediction", pred_color, i + 1, dataformats='CHW')
                                
                                images_logged_this_epoch += 1
                                last_log_batch_idx = val_idx
                                break

            per_class_dice = dice_eval.compute()
            mean_dice = torch.mean(per_class_dice[1:]).item()
            writer.add_scalar("DiceScoreEval/Macro", mean_dice, i + 1)
            writer.add_scalar("DiceScoreEval/Class0", per_class_dice[0], i + 1)
            writer.add_scalar("DiceScoreEval/Class1", per_class_dice[1], i + 1)
            writer.add_scalar("DiceScoreEval/Class2", per_class_dice[2], i + 1)
            writer.add_scalar("DiceScoreEval/Class3", per_class_dice[3], i + 1)
            avg_val_loss = np.mean(val_loss_values)

            # Gets added on top of the Training Loss
            writer.add_scalars("Loss/Total", {'Val':avg_val_loss}, i + 1)
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
    # with pretrain : python -m src.training_scripts.BCP --pretrain_bs 24 --pretrain_epochs 5 --selftrain_bs 16 --selftrain_epochs 5 --pretrained_model_path ./src/pretrained_models/exp_20251016_174938/pretrain_best_model.pth --use_pretrained

    # for local training
    # python -m src.training_scripts.BCP --pretrain_bs 12 --pretrain_epochs 50 --selftrain_bs 12 --selftrain_epochs 50 --optimizer adamw --workers 0


    # python -m src.training_scripts.MT_adaptedlosses --pretrain_bs 12 --pretrain_epochs 10 --optimizer adamw --workers 0 --lmdb_path D:/tfm_data/preprocessed/dataset_lvl4 --model swinunet
    SPLIT = args.split

    # python -m src.training_scripts.MT_balanced --pretrain_bs 10 --pretrain_epochs 25 --optimizer sgd --workers 0 --lmdb_path D:/tfm_data/preprocessed/lmdb_raw --model unet
    if args.hpc:
        output_base = args.output_dir
    else:
        output_base = "./src"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"exp_split{SPLIT}_model{args.model}_{args.pretrain_epochs}epochs_{timestamp}" # f"{args.exp}_pretrepochs{args.pretrain_epochs}_{timestamp}"
    
    
    # experiment_Logs_dir = os.path.join("./src/logs", experiment_name)
    # model_save_dir = os.path.join("./src/pretrained_models", experiment_name)
    experiment_Logs_dir = os.path.join(output_base, "logs", experiment_name)
    clean_path = experiment_Logs_dir.replace("\\", "/")
    model_save_dir = os.path.join(output_base, "pretrained_models", experiment_name)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(experiment_Logs_dir, exist_ok=True)
    logger.info(f"Saving Model Logs to: {clean_path}")


    # Load the split and stuff
    with open("./src/config/splits_20251016_173933.json") as f:
        splits = json.load(f)

    selected_splits = splits.get(f"split_{SPLIT}")
    if selected_splits is None:
        raise ValueError(f"Could not find split {SPLIT} in config file")
    labeled_wsi = selected_splits["labeled_wsi_ids"]
    test_wsi = selected_splits["test_wsi_ids"]
    unlabeled_wsi = selected_splits["unlabeled_wsi_ids"]

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
    #     self_train(args, logs_dir = experiment_Logs_dir, pretrained_model_path=pretrained_model_path, model_save_dir= model_save_dir, labeled_wsi = labeled_wsi, unlabeled_wsi = unlabeled_wsi, test_wsi=test_wsi)
    # else:
    #     if not os.path.isfile(args.pretrained_model_path):
    #         raise FileNotFoundError("Can't find the pretrained model in the specified path")
    #     pretrained_model_path = args.pretrained_model_path
    #     self_train(args, logs_dir=experiment_Logs_dir, pretrained_model_path=args.pretrained_model_path, model_save_dir=model_save_dir, labeled_wsi=labeled_wsi, unlabeled_wsi=unlabeled_wsi, test_wsi=test_wsi)
