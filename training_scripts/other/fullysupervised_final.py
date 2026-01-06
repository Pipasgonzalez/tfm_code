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
from src.utils.data import LMDBTorchDataset, create_balanced_dataloader
from src.helpers import normalize_patch_np, map_seg_to_color_np

COLOR_MAP_NP = np.array([
    [0, 0, 0],       # 0: Background (Black)
    [255, 0, 0],     # 1: Class 1 (Red)
    [0, 255, 0],     # 2: Class 2 (Green)
    [0, 0, 255]      # 3: Class 3 (Blue)
], dtype=np.uint8)

parser = ArgumentParser()

parser.add_argument("--max_epochs", type = int, default = 100)
parser.add_argument("--consistency_rampup", type = float, default = 75.0)
parser.add_argument('--bs',type=int,default=6)
parser.add_argument("--max_iterations", type = int, default = 75)
parser.add_argument("--consistency", type = float, default = 1.0)
parser.add_argument('--hpc', action='store_true')
parser.add_argument("--split",type=int,default=1)
parser.add_argument("--seed",type=float,default=47)
parser.add_argument("--workers",type=int,default=0)
parser.add_argument("--patience", type = int, default = 10)
parser.add_argument("--output_dir", type = str, default="./src")
parser.add_argument("--lr", type = float, default=0.001) # From swin unetr paper it would be 0.0008
args = parser.parse_args()

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def get_current_consistency_weight(current_iteration):
    return args.consistency * sigmoid_rampup(current_iteration, args.consistency_rampup)

# For labeled: All possible ops in one seq, use random_apply to select subset
def apply_labeled_augs_batch(images, targets, ds_type, kornia_all, min_ops=0, max_ops=4):
    # images: (B, C, H, W) GPU tensor, [0,1] float
    # targets: (B, H, W) GPU tensor, long (masks)
    if ds_type != "train":
        return images, targets
    n_ops = random.randint(min_ops, max_ops)
    if n_ops == 0:
        return images, targets
    
    aug_seq = K.AugmentationSequential(
        *kornia_all,  # Your full list of possible ops
        data_keys=["input", "mask"],
        random_apply=n_ops,  # Randomly select and order between min_ops and max_ops ops
        same_on_batch=False,  # Different random params per batch element where supported
        keepdim=True
    )
    augmented = aug_seq(images, targets)
    return augmented[0], augmented[1].squeeze(1)  # [0]=images, [1]=masks

def train_UPC(args, labeled_wsi, val_wsi, logs_dir):
    # tensorboard_path = os.path.join(logs_dir, "pretrain")
    writer = SummaryWriter(logs_dir)
    model_path = os.path.join(logs_dir,"best_upc.pth")
    student = swin_upc()
    meanteacher = MeanTeacher(student_model=student)
    meanteacher.student.cuda()
    meanteacher.teacher.cuda()

    optimizer = torch.optim.AdamW(params = meanteacher.student.parameters(),lr = args.lr,weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=2000, T_mult=2,eta_min=1e-6)
    # warmup = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=int(args.max_iterations)/10)
    # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_iterations, eta_min=1e-6)
    # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[warmup,cosine], milestones=[int(args.max_iterations)/10])

    dice_loss = GeneralizedDiceLoss(include_background=True, to_onehot_y=True, softmax=True, batch=True)
    focal_loss = FocalLoss(include_background=True, to_onehot_y=True,use_softmax=True)

    if not args.hpc:
        dataset_labeled_4 = LMDBTorchDataset("D:/tfm_data/preprocessed/dataset_lvl4_fmt",ds_type='train', allowed_wsi_ids=labeled_wsi)
        dataset_labeled_5 = LMDBTorchDataset("D:/tfm_data/preprocessed/dataset_lvl5_fmt",ds_type='train', allowed_wsi_ids=labeled_wsi)
        dataset_labeled_6 = LMDBTorchDataset("D:/tfm_data/preprocessed/dataset_lvl6",ds_type='train', allowed_wsi_ids=labeled_wsi)
        
        train_dl_lab_4 = create_balanced_dataloader(dataset_labeled_4, num_workers= 0, batch_size=int(args.bs/6), total_images=10, hpc = False)
        train_dl_lab_5 = create_balanced_dataloader(dataset_labeled_5, num_workers= 0, batch_size=int(args.bs/6), total_images=10,hpc =False)
        train_dl_lab_6 = create_balanced_dataloader(dataset_labeled_6, num_workers= 0, batch_size=int(args.bs/6), total_images=10,hpc =False)

        dice_eval = DiceScore(num_classes=4, include_background=True, average='none', input_format='index')
        test_dataset_6 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl6", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_6 = torch.utils.data.DataLoader(dataset=test_dataset_6, batch_size=args.bs, num_workers=0, pin_memory=False, persistent_workers=False)
        test_dataset_4 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl4_fmt", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_4 = torch.utils.data.DataLoader(dataset=test_dataset_4, batch_size=args.bs, num_workers=0, pin_memory=False, persistent_workers=False)
        test_dataset_5 = LMDBTorchDataset(lmdb_path="D:/tfm_data/preprocessed/dataset_lvl5_fmt", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_5 = torch.utils.data.DataLoader(dataset=test_dataset_5, batch_size=args.bs, num_workers=0, pin_memory=False, persistent_workers=False)
    else:
        dataset_labeled_4 = LMDBTorchDataset("./dataset_lvl4",ds_type='train', allowed_wsi_ids=labeled_wsi)
        dataset_labeled_5 = LMDBTorchDataset("./dataset_lvl5",ds_type='train', allowed_wsi_ids=labeled_wsi)
        dataset_labeled_6 = LMDBTorchDataset("./dataset_lvl6",ds_type='train', allowed_wsi_ids=labeled_wsi)
        
        train_dl_lab_4 = create_balanced_dataloader(dataset_labeled_4, num_workers= args.workers, batch_size=int(args.bs/6), hpc = True)
        train_dl_lab_5 = create_balanced_dataloader(dataset_labeled_5, num_workers= args.workers, batch_size=int(args.bs/6), total_images=len(dataset_labeled_4),hpc = True)
        train_dl_lab_6 = create_balanced_dataloader(dataset_labeled_6, num_workers= args.workers, batch_size=int(args.bs/6), total_images=len(dataset_labeled_4),hpc = True)

        dice_eval = DiceScore(num_classes=4, include_background=True, average='none', input_format='index')
        test_dataset_4 = LMDBTorchDataset(lmdb_path="./dataset_lvl4", allowed_wsi_ids=val_wsi, ds_type='test')
        test_dataset_5 = LMDBTorchDataset(lmdb_path="./dataset_lvl5", allowed_wsi_ids=val_wsi, ds_type='test')
        test_dataset_6 = LMDBTorchDataset(lmdb_path="./dataset_lvl6", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_4 = torch.utils.data.DataLoader(dataset=test_dataset_4, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        test_loader_5 = torch.utils.data.DataLoader(dataset=test_dataset_5, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        test_loader_6 = torch.utils.data.DataLoader(dataset=test_dataset_6, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    patches_to_log_per_loader = 4  # Log 4 random images per loader (total up to 12)
    val_losses_per_resolution = {}
    loaders = [
        (test_loader_4, "res4"),
        (test_loader_5, "res5"),
        (test_loader_6, "res6")
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
        meanteacher.student.train()
        meanteacher.teacher.eval()

        # IMPORTANT: Create Train Dataloader Each Time. If not it runs out
        train_labeled_loader = zip(train_dl_lab_4, train_dl_lab_5, train_dl_lab_6)
        for (lab_4, target_4), (lab_5, target_5), (lab_6, target_6) in tqdm(train_labeled_loader,desc=f'Training Epoch {epoch+1}'):
            
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear at start of batch
            if early_stop:
                break
            iter_num +=1
            if iter_num >= max_iterations:
                break
            epoch_train_losses =[]
            
            # Move only labeled to CUDA first
            lab_4, target_4 = lab_4.cuda(non_blocking=True), target_4.cuda(non_blocking=True)
            lab_5, target_5 = lab_5.cuda(non_blocking=True), target_5.cuda(non_blocking=True)
            lab_6, target_6 = lab_6.cuda(non_blocking=True), target_6.cuda(non_blocking=True)
            labeled_images = torch.cat((lab_4, lab_5, lab_6), dim=0)
            labeled_targets = torch.cat((target_4, target_5, target_6), dim=0)

            # Aug labeled
            labeled_images, labeled_targets = apply_labeled_augs_batch(
                labeled_images, labeled_targets, ds_type='train', 
                kornia_all=dataset_labeled_4.kornia_all,
                min_ops=0, max_ops=dataset_labeled_4.nr_labeled_transforms
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Supervised forward/loss
                out_supervised = meanteacher.student(labeled_images)
                dice_loss_supervised = dice_loss(out_supervised, labeled_targets.unsqueeze(1))
                focal_loss_supervised = focal_loss(out_supervised, labeled_targets.unsqueeze(1))
                supervised_loss = dice_loss_supervised + focal_loss_supervised

            # Delete labeled tensors immediately (free ~half the input memory)
            del labeled_images, labeled_targets, out_supervised, lab_4, lab_5, lab_6, target_4, target_5, target_6
            torch.cuda.empty_cache()  # Force release
            
            supervised_loss.backward()
            optimizer.step()
            meanteacher.update_teacher()
            lr_scheduler.step()

            writer.add_scalars('loss/supervised',{'dice':dice_loss_supervised.item(),'focal':focal_loss_supervised.item()}, iter_num)

            # Logging Learning Rate
            writer.add_scalar("params/learning_rate", optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('params/consistency_weight', get_current_consistency_weight(current_iteration=iter_num), iter_num)
            epoch_train_losses.append(supervised_loss.item())

            if iter_num % 250 == 0:
                writer.add_scalars('loss/total', {'train':np.mean(epoch_train_losses)}, iter_num)

                ##### VALIDATION LOOP

                with torch.no_grad():
                    dice_eval.reset()
                    dice_score_per_resolution = {}

                    for loader, res_tag in loaders:
                        val_loss_values = []
                        # New Dice Score for each resolution
                        dice_eval.reset()
                        
                        log_indices_set = test_info[res_tag]["indices"]
                        num_batches = test_info[res_tag]["num_batches"]
                        progress_bar_val = tqdm(loader, 
                                                desc=f"Running Validation on Iter Num: {iter_num} ({res_tag})", 
                                                total=num_batches)

                        for val_idx, batch in enumerate(progress_bar_val):
                            images, targets, wsi_ids = batch
                            images, targets = images.cuda(), targets.cuda()
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                out = meanteacher.teacher(images)
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
                    torch.save(meanteacher.teacher.state_dict(), model_path)
                else:
                    patience_counter +=1
                
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered")
                    early_stop = True
                    break
                meanteacher.student.train()

        # Final Breakout
        if iter_num >= max_iterations:
            break # BREAK out of the outer training loop aswell
        if early_stop:
            break
            
if __name__ == "__main__":
    # python -m src.training_scripts.UPC --bs 12
    if args.hpc:
        output_base = args.output_dir
    else:
        output_base = "./src"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CHANGE sortable experiment name
    experiment_name = f"{timestamp}_split{args.split}_{args.max_epochs}epochs_{args.bs}bs"
    experiment_Logs_dir = os.path.join(output_base, "logs", experiment_name)
    clean_path = experiment_Logs_dir.replace("\\", "/")
    os.makedirs(experiment_Logs_dir, exist_ok=True)


    # Load the split and stuff
    with open("./src/config/splits_3110.json") as f:
        splits = json.load(f)
    selected_splits = splits.get(f"split_{args.split}")
    if selected_splits is None:
        raise ValueError(f"Could not find split {args.split} in config file")
    labeled_wsi = selected_splits["labeled_wsi_ids"]
    test_wsi = selected_splits["test_wsi_ids"]
    
    with open('./src/config/unlabeled_images.json') as f:
        unlabeled_wsi_config = json.load(f)
    
    unlabeled_wsi_ids = []
    for unlabeled_group in unlabeled_wsi_config.values():
        unlabeled_wsi_ids.extend(unlabeled_group)

    torch.backends.cudnn.benchmark = True # Can be changed to false
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(clean_path)
    train_UPC(args,labeled_wsi=labeled_wsi,val_wsi=test_wsi,unlabeled_wsi=unlabeled_wsi_ids,logs_dir=experiment_Logs_dir)

