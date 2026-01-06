############ ABLATION STUDY 1.1.1: Completely Random Patch Sampling of Patches from ONE Resolution


import torch
import torch.nn.functional as f
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from monai.losses.dice import GeneralizedDiceLoss
from monai.losses.focal_loss import FocalLoss
import torchvision.transforms.v2 as v2
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.segmentation import DiceScore, GeneralizedDiceScore

import itertools
import os
from loguru import logger
import numpy as np
from datetime import datetime
import random
import json
from pprint import pprint
import time
import kornia.augmentation as K

from src.networks.swin_unet import swin_upc
from src.networks.ema import MeanTeacher
from src.utils.dataset_supervised import LMDBTorchDataset, create_balanced_dataloader
from src.helpers import normalize_patch_np, map_seg_to_color_np

COLOR_MAP_NP = np.array([
    [0, 0, 0],       # 0: Background (Black)
    [255, 0, 0],     # 1: Class 1 (Red)
    [0, 255, 0],     # 2: Class 2 (Green)
    [0, 0, 255]      # 3: Class 3 (Blue)
], dtype=np.uint8)

parser = ArgumentParser()

parser.add_argument("--max_epochs", type = int, default = 10000)
parser.add_argument('--bs',type=int,default=16)
parser.add_argument("--max_iterations", type = int, default = 10001)
parser.add_argument('--hpc', action='store_true')
parser.add_argument("--split",type=int,default=1)
parser.add_argument("--seed",type=float,default=47)
parser.add_argument("--workers",type=int,default=0)
parser.add_argument("--patience", type = int, default = 100)
parser.add_argument("--output_dir", type = str, default="./src")
parser.add_argument("--lr", type = float, default=5e-4) # From swin unetr paper it would be 0.0008
args = parser.parse_args()

def train_supervised(args, labeled_wsi, val_wsi, logs_dir):
    # tensorboard_path = os.path.join(logs_dir, "pretrain")
    writer = SummaryWriter(logs_dir)
    model_path = os.path.join(logs_dir,"best_supervised.pth")
    model_path_last = os.path.join(logs_dir,"last_model.pth")
    model = swin_upc()
    model.to("cuda")

    optimizer = torch.optim.AdamW(params = model.parameters(),lr = args.lr,weight_decay=5e-4)

    dice_loss = GeneralizedDiceLoss(include_background=True, to_onehot_y=True, softmax=True, batch=True)
    focal_loss = FocalLoss(include_background=True, to_onehot_y=True,use_softmax=True)
    dice_eval = DiceScore(num_classes=4, include_background=True, average='none', input_format='index')
    if not args.hpc:
        dataset_labeled_high = LMDBTorchDataset("D:/tfm_data/preprocessed/dataset_lvl3_fmt",ds_type='train', allowed_wsi_ids=labeled_wsi, included_sampling_categories=[0,1]) # 2 are the boring patches
        train_dl_lab_high = create_balanced_dataloader(dataset_labeled_high,num_workers=args.workers, batch_size=args.bs, hpc = True)
        dataset_labeled_medium = LMDBTorchDataset("./dataset_lvl4",ds_type='train', allowed_wsi_ids=labeled_wsi, included_sampling_categories= [0,1])
        train_dl_lab_medium = create_balanced_dataloader(dataset_labeled_medium, num_workers= args.workers, batch_size=args.bs, hpc = True, total_images=len(dataset_labeled_high))
        dataset_labeled_low = LMDBTorchDataset("./dataset_lvl5",ds_type='train', allowed_wsi_ids=labeled_wsi, included_sampling_categories= [0,1])
        train_dl_lab_low = create_balanced_dataloader(dataset_labeled_low, num_workers= args.workers, batch_size=args.bs, hpc = True, total_images=len(dataset_labeled_high))
        
        test_dataset_high = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl3_fmt", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_high = torch.utils.data.DataLoader(dataset=test_dataset_high, batch_size=4, num_workers=0, pin_memory=False, persistent_workers=False)
        test_dataset_medium = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl4_fmt", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_medium = torch.utils.data.DataLoader(dataset=test_dataset_medium, batch_size=4, num_workers=0, pin_memory=False, persistent_workers=False)
        test_dataset_low = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl5_fmt", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_low = torch.utils.data.DataLoader(dataset=test_dataset_low, batch_size=4, num_workers=0, pin_memory=False, persistent_workers=False)
    else:
        dataset_labeled_high = LMDBTorchDataset("./dataset_lvl3",ds_type='train', allowed_wsi_ids=labeled_wsi, included_sampling_categories= [0,1])
        train_dl_lab_high = create_balanced_dataloader(dataset_labeled_high, num_workers= args.workers, batch_size=6, hpc = True)
        dataset_labeled_medium = LMDBTorchDataset("./dataset_lvl4",ds_type='train', allowed_wsi_ids=labeled_wsi, included_sampling_categories= [0,1])
        train_dl_lab_medium = create_balanced_dataloader(dataset_labeled_medium, num_workers= args.workers, batch_size=6, hpc = True, total_images=len(dataset_labeled_high))
        dataset_labeled_low = LMDBTorchDataset("./dataset_lvl5",ds_type='train', allowed_wsi_ids=labeled_wsi, included_sampling_categories= [0,1])
        train_dl_lab_low = create_balanced_dataloader(dataset_labeled_low, num_workers= args.workers, batch_size=4, hpc = True, total_images=len(dataset_labeled_high))

        test_dataset_high = LMDBTorchDataset(lmdb_path="./dataset_lvl3", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_high = torch.utils.data.DataLoader(dataset=test_dataset_high, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        test_dataset_medium = LMDBTorchDataset(lmdb_path="./dataset_lvl4", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_medium = torch.utils.data.DataLoader(dataset=test_dataset_medium, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        test_dataset_low = LMDBTorchDataset(lmdb_path="./dataset_lvl5", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_low = torch.utils.data.DataLoader(dataset=test_dataset_low, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    logger.info(f"Length of the Training Dataset on Highest Resolution: {len(dataset_labeled_high)}")
    logger.info(f"Length of the Training Dataset on Medium Resolution: {len(dataset_labeled_medium)}")
    logger.info(f"Length of the Test Dataset on Highest Resolution: {len(test_dataset_high)}")
    logger.info(f"Length of the Test Dataset on Medium Resolution: {len(test_dataset_medium)}")

    patches_to_log_per_loader = 4  # Log 4 random images per loader (total up to 12)
    val_losses_per_resolution = {}
    loaders = [
        (test_loader_high, "res3"),
        (test_loader_medium, "res4"),
        (test_loader_low, "res5")
    ]
    test_info = {}
    for (loader,res_tag) in loaders:
        num_batches = len(loader)
        if num_batches < patches_to_log_per_loader:
            log_indices_set = set(range(num_batches))
        else:
            log_indices = np.random.choice(num_batches, patches_to_log_per_loader, replace=False)
            log_indices_set = set(log_indices)
        test_info[res_tag] = {"indices":log_indices_set, "num_batches":num_batches}

    max_iterations = args.max_iterations
    iter_num = 0
    best_performance = 0
    patience_counter = 0
    early_stop = False

    writer.add_hparams(
        {
            "batch_size":args.bs,
            "max_iterations":args.max_iterations,
            "optimizer":type(optimizer).__name__,
            "learning_rate":args.lr,
        },
        {"dummy": 0}
    )

    for epoch in range(args.max_epochs):
        model.train()
        train_labeled_loader = zip(train_dl_lab_high, train_dl_lab_medium, train_dl_lab_low)
        for (lab_high, target_high), (lab_medium, target_medium), (lab_low, target_low) in tqdm(train_labeled_loader,desc=f'Training Epoch {epoch+1}'):
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear at start of batch
            if early_stop:
                break
            iter_num +=1
            if iter_num >= max_iterations:
                break
            epoch_train_losses =[]
            
            # Move only labeled to CUDA first
            lab_high, target_high = lab_high.cuda(non_blocking=True), target_high.cuda(non_blocking=True)
            lab_medium, target_medium = lab_medium.cuda(non_blocking=True), target_medium.cuda(non_blocking=True)
            lab_low, target_low = lab_low.cuda(non_blocking = True), target_low.cuda(non_blocking = True)

            labeled_images = torch.cat((lab_high, lab_medium, lab_low), dim=0)
            labeled_targets = torch.cat((target_high, target_medium, target_low), dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Supervised forward/loss
                out_supervised = model(labeled_images)
                dice_loss_supervised = dice_loss(out_supervised, labeled_targets.unsqueeze(1))
                focal_loss_supervised = focal_loss(out_supervised, labeled_targets.unsqueeze(1))
                supervised_loss = dice_loss_supervised + focal_loss_supervised
            
            supervised_loss.backward()
            optimizer.step()

            writer.add_scalars('loss/supervised',{'dice':dice_loss_supervised.item(),'focal':focal_loss_supervised.item()}, iter_num)
            writer.add_scalar("params/learning_rate", optimizer.param_groups[0]['lr'], iter_num)
            epoch_train_losses.append(supervised_loss.item())

            if iter_num % 500 == 0:
                logger.info(f"Running Validation at Iter:{iter_num} ")
                writer.add_scalars('loss/total', {'train':np.mean(epoch_train_losses)}, iter_num)
                model.eval()
                ##### VALIDATION LOOP
                with torch.no_grad():
                    dice_eval.reset()
                    dice_score_per_resolution = {}

                    for loader, res_tag in loaders:
                        val_loss_values = []
                        dice_eval.reset()
                        
                        log_indices_set = test_info[res_tag]["indices"]
                        num_batches = test_info[res_tag]["num_batches"]
                        # progress_bar_val = tqdm(loader, 
                        #                         desc=f"Running Validation on Iter Num: {iter_num} ({res_tag})", 
                        #                         total=num_batches)

                        for val_idx, batch in enumerate(loader):
                            images, targets, wsi_ids = batch
                            images, targets = images.cuda(), targets.cuda()
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                out = model(images)
                                dice_loss_eval = dice_loss(out, targets.unsqueeze(1))
                                focal_loss_eval = focal_loss(out, targets.unsqueeze(1))
                                total_loss_eval = focal_loss_eval + dice_loss_eval
                                pred_prob_eval = torch.nn.functional.softmax(out, dim=1)
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
                                wsi_id = wsi_ids[b]
                                # 1. Normalize original image (NumPy)
                                img_normalized = normalize_patch_np(image_patch_np)
                                # 2. Color-map ground truth (NumPy)
                                target_color = map_seg_to_color_np(target_patch_np, COLOR_MAP_NP)
                                # 3. Color-map prediction (NumPy)
                                pred_color = map_seg_to_color_np(pred_patch_np, COLOR_MAP_NP)
                                # 4. Log to TensorBoard
                                tag_prefix = f"val/res{res_tag}_batch{wsi_id}"
                                writer.add_image(f"{tag_prefix}_Image", img_normalized, iter_num, dataformats='CHW')
                                writer.add_image(f"{tag_prefix}_GroundTruth", target_color, iter_num, dataformats='CHW')
                                writer.add_image(f"{tag_prefix}_Prediction", pred_color, iter_num, dataformats='CHW')

                        per_class_dice = dice_eval.compute()
                        mean_dice = torch.mean(per_class_dice[1:]).item()
                        dice_score_per_resolution[res_tag] = mean_dice
                        dice_dict = {
                            'Macro': mean_dice,
                            'Class0': per_class_dice[0].item(),
                            'Class1': per_class_dice[1].item(),
                            'Class2': per_class_dice[2].item(),
                            'Class3': per_class_dice[3].item()
                        }
                        writer.add_scalars(f"val/dsc_{res_tag}", dice_dict, iter_num)
                        val_losses_per_resolution[res_tag] = np.mean(val_loss_values)

                    # Gets added on top of the Training Loss
                    global_loss = np.mean([np.mean(val_losses) for val_losses in val_losses_per_resolution.values()])
                    writer.add_scalars("loss/total", {"val":global_loss}, iter_num)

                global_dice = np.mean([np.mean(dice_score) for dice_score in dice_score_per_resolution.values()])
                if global_dice > best_performance:
                    best_performance = global_dice
                    patience_counter = 0
                    logger.info(f"Saving New Best Pre-Train Model with Dice Score of {global_dice}")
                    torch.save(model.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path_last)
                    patience_counter +=1
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered")
                    early_stop = True
                    break
                model.train()

        # Final Breakout
        if iter_num >= max_iterations:
            break
        if early_stop:
            break
            
if __name__ == "__main__":
    if args.hpc:
        output_base = args.output_dir
    else:
        output_base = "./src"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = f"{timestamp}_split{args.split}_{args.max_iterations}iterations_{args.bs}bs"
    experiment_Logs_dir = os.path.join(output_base, "logs", experiment_name)
    clean_path = experiment_Logs_dir.replace("\\", "/")
    os.makedirs(experiment_Logs_dir, exist_ok=True)

    # Load the split and stuff
    with open("./src/config/splits_faber.json") as f:
        splits = json.load(f)
    selected_splits = splits.get(f"split_{args.split}")
    if selected_splits is None:
        raise ValueError(f"Could not find split {args.split} in config file")
    labeled_wsi = selected_splits["labeled_wsi_ids"]
    test_wsi = selected_splits["test_wsi_ids"]

    torch.backends.cudnn.benchmark = True # Can be changed to false
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(clean_path)
    train_supervised(args,labeled_wsi=labeled_wsi,val_wsi=test_wsi,logs_dir=experiment_Logs_dir)

