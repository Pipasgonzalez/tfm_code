import argparse
import sys

from monai.networks.nets.basic_unet import BasicUNet
from monai.losses.dice import DiceCELoss, DiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.losses.tversky import TverskyLoss
from monai.losses.dice import DiceFocalLoss


from src.utils.data import LMDBTorchDataset, create_balanced_dataloader
from src.networks.unet import get_model
from src.networks.swin_unet import get_swinunet
from src.networks.ema import MeanTeacher
from src.utils.losses import AsymmetricUnifiedFocalLoss
# from src.utils.losses import DiceLoss

import numpy as np
import random

from torch.utils.data import Subset, Dataset, DataLoader
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
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--weight_decay", type=float, default = 1e-4)
parser.add_argument("--pretrain_bs", type=int, default=12)
parser.add_argument("--selftrain_bs", type = int, default = 12)
parser.add_argument("--pretrain_epochs", type=int, default=100)
parser.add_argument("--selftrain_epochs", type=int, default = 100)
parser.add_argument("--use_pretrained", action="store_true", help="Whether or not to use the pretrained model for the self training part")
parser.add_argument("--pretrained_model_path", type=str, default = "./src/pretrained_models/exp_20251015_170238/pretrain_best_model.pth")
parser.add_argument("--split", type=int, default=3)
parser.add_argument("--patience", type=int,default=10)
parser.add_argument("--workers", type=int, default = 4)
parser.add_argument('--multiplier_unet', type = int, default=1)
parser.add_argument('--hpc', action='store_true')
parser.add_argument('--output_dir', type=str, default='./src', help='Base directory for outputs (used with --hpc)')
args = parser.parse_args()

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
    cat1_idx = [idx for idx, metdat in enumerate(train_dataset.metadata) if train_dataset.labeled_categories[idx][1]==0]
    cat2_idx = [idx for idx, metdat in enumerate(train_dataset.metadata) if train_dataset.labeled_categories[idx][1]==1]
    cat3_idx = [idx for idx, metdat in enumerate(train_dataset.metadata) if train_dataset.labeled_categories[idx][1]==2]
    cat1_ds = Subset(train_dataset, indices=cat1_idx)
    cat2_ds = Subset(train_dataset, indices=cat2_idx)
    cat3_ds = Subset(train_dataset, indices=cat3_idx)
    cat1_dl = DataLoader(dataset=cat1_ds, batch_size=8, shuffle=True,num_workers=workers, pin_memory=True)
    cat2_dl = DataLoader(dataset=cat2_ds, batch_size=2, shuffle=True,num_workers=workers, pin_memory=True)
    cat3_dl = DataLoader(dataset=cat3_ds, batch_size=2, shuffle=True,num_workers=workers, pin_memory=True)


    # train_loader = create_balanced_dataloader(train_dataset, batch_size=bs, num_workers=workers, replacement=True)
    test_dataset = LMDBTorchDataset(lmdb_path=lmdb_path, allowed_wsi_ids=test_wsi_ids, ds_type='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=workers, pin_memory=True)

    if model_type == 'unet':
        model = get_model(nclasses=num_classes, device="cuda", multiplier = multiplier)
    elif model_type == 'swinunet':
        model = get_swinunet()
    else:
        raise ValueError(f'No implementation currently for Network Type {model}')

    if optimizer == "sgd":
        optimizer = torch.optim.SGD(params=model.parameters(),lr = 0.01, momentum=0.9,weight_decay=0.0001) # Directly from the original paper
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr,weight_decay=weight_decay # momentum=0.9, weight_decay=0.0001
        )
    else:
        raise ValueError("Optimizer has to be either sgd or adamw")


    # CHANGE add weights to CE
    ce_weights = torch.from_numpy(np.array([0.8, 0.1, 0.1])) # from class_distributions.ipynb
    # unified_loss = AsymmetricUnifiedFocalLoss(lambda_weight=0.5,delta=0.6,gamma=0.5,num_classes=4,rare_class_idx=1,background_class_idx=0)

    foc_loss = FocalLoss(include_background=False,to_onehot_y=True,use_softmax=True,weight=ce_weights) # CHANGE removed weights
    tvsky_loss = TverskyLoss(include_background=False,to_onehot_y=True,softmax=True,alpha=0.2,beta=0.8)

    # New DiceFocal
    dab_loss = DiceFocalLoss(include_background=False,to_onehot_y=True,softmax=True,lambda_dice=2, lambda_focal=4, weight=[0.8, 0.1, 0.1])

    # dice_eval = DiceScore(num_classes=4, include_background=False, input_format="index", average="macro")
    dice_eval = DiceScore(num_classes=4, include_background=True, input_format="index", average="none") # This version aggregated dice scores per class to log them out properly
    # Scaler
    scaler = GradScaler(device="cuda",)  # manages scaling automatically

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

    STATIC_FOCAL_SCALE = 20
    for i in range(max_epoch):

        # Adaptive weighting variable
        # alpha will ramp from 0.0 (epoch 0) to 1.0 (final epoch)
        alpha = i / (max_epoch - 1) if max_epoch > 1 else 0.0 
        
        lambda_tversky = 1.0 - alpha + 0.05  # Starts at 1.0, ends at 0.0
        lambda_focal = alpha + 0.05         # Starts at 0.0, ends at 1.0
            # Log the weights themselves to see them ramp
        writer.add_scalar("Loss/Pretrain/Lambda_Focal", lambda_focal, iter_num)
        writer.add_scalar("Loss/Pretrain/Lambda_Tversky", lambda_tversky, iter_num)
        model.cuda()
        model.train()

        # progress_bar_train = tqdm(train_loader,desc=f"Epoch {i+1}/{max_epoch}")
        progress_bar_train = tqdm(zip(cat1_dl, cat2_dl, cat3_dl), desc=f"Epoch {i+1}/{max_epoch}",total=min(len(cat1_dl),len(cat2_dl), len(cat3_dl)))
        for (patch1, target1), (patch2, target2), (patch3, target3) in progress_bar_train:
        # for (patch1, target1), (patch2, target2) in progress_bar_train:
            optimizer.zero_grad()

            # Concatenate across dataloaders (along batch dimension)
            patch = torch.cat([patch1, patch2, patch3], dim=0).cuda(non_blocking=True)
            target = torch.cat([target1, target2, target3], dim=0).cuda(non_blocking=True)

            with autocast(device_type="cuda",dtype=torch.float16):
                out= model(patch)
                # total_unified_loss,loss_tversky, loss_focal = unified_loss(out,target)
                focal_loss_batch = foc_loss(out,target.unsqueeze(1)) * STATIC_FOCAL_SCALE
                tversky_loss_batch = tvsky_loss(out,target.unsqueeze(1))
                total_loss = lambda_focal * focal_loss_batch + lambda_tversky * tversky_loss_batch

                dab_loss_value = dab_loss(out,target.unsqueeze(1))
            # Training Loop
            scaler.scale(dab_loss_value).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()

            iter_num += 1
            if iter_num % 50 == 0:
                with torch.no_grad():
                    
                    # --- For the combined batch (what the previous code did) ---
                    total_pixels = target.numel()
                    bincounts = torch.bincount(target.flatten().cpu(), minlength=num_classes)
                    percentages = (bincounts.float() / total_pixels) * 100.0
                    dist_str = ", ".join([f"Class {i}: {p:.2f}%" for i, p in enumerate(percentages)])
                    logger.info(f"Iter {iter_num} - COMBINED Batch Dist: [{dist_str}]")

                    # --- For the separate dataloaders ---
                    targets_list = [target1, target2, target3]
                    dl_names = ["Cat1_DL (8)", "Cat2_DL (2)", "Cat3_DL (2)"] # (4), (2) are your batch sizes

                    for name, t in zip(dl_names, targets_list):
                        # Make sure the tensor isn't empty (though it shouldn't be)
                        if t.numel() > 0: 
                            total_pixels_dl = t.numel()
                            bincounts_dl = torch.bincount(t.flatten().cpu(), minlength=num_classes)
                            percentages_dl = (bincounts_dl.float() / total_pixels_dl) * 100.0
                            dist_str_dl = ", ".join([f"Class {i}: {p:.2f}%" for i, p in enumerate(percentages_dl)])
                            logger.info(f"Iter {iter_num} - {name} Batch Dist: [{dist_str_dl}]")
                    
                    # Add a separator line for readability in the logs
                    logger.info("-" * 20)

            writer.add_scalar("Loss/Pretrain/Total", dab_loss_value.item(), iter_num)
            
            # --- ADD THESE LINES ---
            # Log the raw, un-weighted loss values
            writer.add_scalar("Loss/Pretrain/Raw_Focal", focal_loss_batch.item(), iter_num)
            writer.add_scalar("Loss/Pretrain/Raw_Tversky", tversky_loss_batch.item(), iter_num)
            
            
            # Log the final, weighted contribution of each loss
            writer.add_scalar("Loss/Pretrain/Weighted_Focal_Contribution", (lambda_focal * focal_loss_batch).item(), iter_num)
            writer.add_scalar("Loss/Pretrain/Weighted_Tversky_Contribution", (lambda_tversky * tversky_loss_batch).item(), iter_num)
            progress_bar_train.set_postfix({"Batch Loss": f"{total_loss.item():.4f}"})
        # Begin Eval
        model.eval()

        # To avoid gradient calculation etc.
        if not skip_validation:
            with torch.no_grad():
                dice_eval.reset()

                dice_scores = []  # store all batch Dice scores
                progress_bar_val = tqdm(test_loader, desc=f"Running Validation on Epoch: {i+1}")
                for batch in progress_bar_val:
                    eval_num += 1
                    images, targets = batch
                    images, targets = images.cuda(), targets.cuda()
                    with autocast(device_type="cuda", dtype=torch.float16):
                        out = model(images)
                        pred_prob = F.softmax(out, dim=1)
                        pred = torch.argmax(pred_prob, dim=1)

                    # dice_score = dice_eval(pred, targets) DEPRECATED
                    dice_eval.update(pred, targets)
                    per_class_dice = dice_eval.compute()
                    mean_dice = torch.mean(per_class_dice[1:]).item()

                    # dice_scores.append(dice_score.item())  # move to CPU float
                    progress_bar_val.set_postfix({"Batch Dice": f"{mean_dice:.4f}"})
            per_class_dice = dice_eval.compute()
            mean_dice = torch.mean(per_class_dice[1:]).item()
            writer.add_scalar("DiceScoreEval/Macro", mean_dice, i + 1)
            # writer.add_scalar("DiceScoreEval/Class0", per_class_dice[0], i + 1)
            writer.add_scalar("DiceScoreEval/Class1", per_class_dice[1], i + 1)
            writer.add_scalar("DiceScoreEval/Class2", per_class_dice[2], i + 1)
            writer.add_scalar("DiceScoreEval/Class3", per_class_dice[3], i + 1)
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


    if args.hpc:
        output_base = args.output_dir
    else:
        output_base = "./src"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"exp_split{SPLIT}_{args.pretrain_epochs}epochs_{timestamp}" # f"{args.exp}_pretrepochs{args.pretrain_epochs}_{timestamp}"
    
    
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
