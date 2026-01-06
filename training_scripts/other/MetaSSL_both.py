import torch
import torch.nn.functional as F
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
from src.utils.dataset_unlabeled import LMDBTorchDatasetUnlabeled
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
parser.add_argument("--consistency", type = float, default = 5.0)
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
    
def get_current_consistency_weight(current_iteration, rampup_length,max_consistency):
    return max_consistency * sigmoid_rampup(current_iteration, rampup_length)

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

# For unlabeled: Separate weak and strong seqs
def apply_unlabeled_augs_batch(images, kornia_weak, kornia_strong, min_weak_ops=0, max_weak_ops=2, min_strong_ops=1, max_strong_ops=2):
    # images: (B, C, H, W) GPU tensor, [0,1] float
    n_ops_weak = random.randint(min_weak_ops, max_weak_ops)
    if n_ops_weak != 0:
        # Weak: Batched apply to full tensor
        weak_seq = K.AugmentationSequential(
            *kornia_weak,  # All weak ops
            data_keys=["input"],
            random_apply=n_ops_weak,
            same_on_batch=False,
            keepdim=True
        )
        weak_images = weak_seq(images)
    else:
        weak_images = images
    
    # Strong: Apply to clone of weak (batched)
    strong_seq = K.AugmentationSequential(
        *kornia_strong,  # All strong ops
        data_keys=["input"],
        random_apply=(min_strong_ops, max_strong_ops),
        same_on_batch=False,
        keepdim=True
    )
    noisy_images = strong_seq(weak_images.clone())  # Or images.clone() if strong shouldn't chain on weak
    
    return weak_images, noisy_images


def generalized_gaussian(u, beta=3.0):
    return torch.exp(-torch.pow(u, beta))


def calculate_weighted_dice_loss(p_soft, y_true_onehot, weight_map, num_classes=4):
    """
    Implements the MetaSSL weighted Dice loss (Eq. 8).
    p_soft: Student's softmax output [B, C, H, W]
    y_true_onehot: Ground truth one-hot [B, C, H, W]
    weight_map: Pixel-wise weights [B, H, W]
    """
    C = num_classes
    # Add channel dim to weight_map for broadcasting
    weight_map_c = weight_map.unsqueeze(1) # [B, 1, H, W]
    
    # Eq. 8 numerator, summed per class [cite: 264]
    numerator_per_class = torch.sum(
        weight_map_c * 2.0 * y_true_onehot * p_soft, 
        dim=(0, 2, 3) # Sum over batch and spatial dims
    )
    
    # Eq. 8 denominator, summed per class [cite: 264]
    denominator_per_class = torch.sum(
        weight_map_c * (y_true_onehot + p_soft), 
        dim=(0, 2, 3) # Sum over batch and spatial dims
    )
    
    # Eq. 8 final formula [cite: 264]
    dice_per_class = (numerator_per_class + 1e-5) / (denominator_per_class + 1e-5)
    
    # We average over all classes, including background, as in the paper's formula
    return 1.0 - torch.mean(dice_per_class)

def train_UPC(args, labeled_wsi, val_wsi, unlabeled_wsi, logs_dir):
    unc_per_wsi = defaultdict(list)
    # tensorboard_path = os.path.join(logs_dir, "pretrain")
    writer = SummaryWriter(logs_dir)
    model_path = os.path.join(logs_dir,"best_upc.pth")
    student = swin_upc()
    meanteacher = MeanTeacher(student_model=student)
    meanteacher.student.cuda()
    meanteacher.teacher.cuda()

    optimizer = torch.optim.SGD(params = meanteacher.student.parameters(),lr = 0.01,momentum=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=2000, T_mult=2,eta_min=1e-6)
    # warmup = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=int(args.max_iterations)/10)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_iterations, eta_min=1e-4)
    # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[warmup,cosine], milestones=[int(args.max_iterations)/10])

    scaler = torch.GradScaler(device="cuda")
    dice_loss = GeneralizedDiceLoss(include_background=True, to_onehot_y=True, softmax=True, batch=True)
    focal_loss = FocalLoss(include_background=True, to_onehot_y=True,use_softmax=True)

    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    sm = torch.nn.Softmax(dim=1)
    logsm = torch.nn.LogSoftmax(dim=1)
    kl_distance = torch.nn.KLDivLoss(reduction='none')

    if not args.hpc:
        dataset_labeled_4 = LMDBTorchDataset("D:/tfm_data/preprocessed/dataset_lvl4_fmt",ds_type='train', allowed_wsi_ids=labeled_wsi)
        dataset_labeled_5 = LMDBTorchDataset("D:/tfm_data/preprocessed/dataset_lvl5_fmt",ds_type='train', allowed_wsi_ids=labeled_wsi)
        dataset_labeled_6 = LMDBTorchDataset("D:/tfm_data/preprocessed/dataset_lvl6",ds_type='train', allowed_wsi_ids=labeled_wsi)
        
        train_dl_lab_4 = create_balanced_dataloader(dataset_labeled_4, num_workers= 0, batch_size=int(args.bs/6), total_images=10, hpc = False)
        train_dl_lab_5 = create_balanced_dataloader(dataset_labeled_5, num_workers= 0, batch_size=int(args.bs/6), total_images=10,hpc =False)
        train_dl_lab_6 = create_balanced_dataloader(dataset_labeled_6, num_workers= 0, batch_size=int(args.bs/6), total_images=10,hpc =False)

        dataset_unlabeled = LMDBTorchDataset("D:/tfm_data/preprocessed/dataset_unlabeled_lvl5", ds_type='train',allowed_wsi_ids=unlabeled_wsi,unlabeled=True,return_wsi_id=True) # CHANGE returning whole slide images now
        unlabeled_sampler = torch.utils.data.RandomSampler(dataset_unlabeled,replacement=True,num_samples=len(dataset_labeled_4))
        train_dl_unlab = torch.utils.data.DataLoader(dataset_unlabeled,batch_size=int(args.bs/2), num_workers= 0,pin_memory=False,drop_last=True,persistent_workers=False,sampler=unlabeled_sampler)

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

        dataset_unlabeled = LMDBTorchDatasetUnlabeled("./dataset_unlabeled_lvl5",allowed_wsi_ids=unlabeled_wsi,return_wsi_id=True) # CHANGE returning whole slide images now
        unlabeled_sampler = torch.utils.data.RandomSampler(dataset_unlabeled,replacement=True,num_samples=len(dataset_labeled_4))
        train_dl_unlab = torch.utils.data.DataLoader(dataset_unlabeled,batch_size=int(args.bs/2), num_workers= args.workers if args.hpc else 0,pin_memory=True,drop_last=True,persistent_workers=True,sampler=unlabeled_sampler)

        dice_eval = DiceScore(num_classes=4, include_background=True, average='none', input_format='index')
        test_dataset_4 = LMDBTorchDataset(lmdb_path="./dataset_lvl4", allowed_wsi_ids=val_wsi, ds_type='test')
        test_dataset_5 = LMDBTorchDataset(lmdb_path="./dataset_lvl5", allowed_wsi_ids=val_wsi, ds_type='test')
        test_dataset_6 = LMDBTorchDataset(lmdb_path="./dataset_lvl6", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_4 = torch.utils.data.DataLoader(dataset=test_dataset_4, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        test_loader_5 = torch.utils.data.DataLoader(dataset=test_dataset_5, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        test_loader_6 = torch.utils.data.DataLoader(dataset=test_dataset_6, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)


    logger.info(f"Length of Unlabeled Dataset at batchsize {args.bs}: {len(dataset_unlabeled)}\n Length of Highest resolution labeled Dataset : {len(dataset_labeled_4)}. \n Middle Resolution: {len(dataset_labeled_5)} \n Lowest Resolution: {len(dataset_labeled_6)}")

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


    RAMPUP_PERCENTAGE = 0.5
    max_iterations = args.max_iterations
    rampup_iterations = int(max_iterations * RAMPUP_PERCENTAGE)
    args.consistency_rampup = rampup_iterations
    max_consistency = args.consistency
    iter_num = 0
    best_performance = 0
    patience_counter = 0
    early_stop = False
    # Add hyperparameters to the tensorboard


    # MetaSSL
    initial_threshold = 0.5 
    # Use a dictionary for clarity
    class_thresholds_g = {
        c: torch.tensor(initial_threshold, device="cuda", dtype=torch.float) 
        for c in range(4)
    }
    # EMA momentum for threshold updates (paper's code uses 0.5, but you can tune)
    threshold_ema_alpha = 0.5


    writer.add_hparams(
        {
            "batch_size":args.bs,
            "max_iterations":args.max_iterations,
            "rampup_percentage":RAMPUP_PERCENTAGE,
            "optimizer":type(optimizer).__name__,
            "learning_rate":args.lr,
            "size_unlabeled":len(dataset_unlabeled),
            "size_largest_unlabeled": len(dataset_labeled_4),
            "consistency":args.consistency
        },
        {"dummy": 0}
    )

    for epoch in range(args.max_epochs):
        meanteacher.student.train()
        meanteacher.teacher.eval()

        # IMPORTANT: Create Train Dataloader Each Time. If not it runs out
        train_labeled_loader = zip(train_dl_lab_4, train_dl_lab_5, train_dl_lab_6)
        for (((lab_4, target_4), (lab_5, target_5), (lab_6, target_6)),(unlab_images, unlab_wsi_ids)) in tqdm(zip(train_labeled_loader,train_dl_unlab),desc=f'Training Epoch {epoch+1}', total=min(len(train_dl_lab_4),len(train_dl_unlab))):
            
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
                out_supervised_student = meanteacher.student(labeled_images) # 'p'
                with torch.no_grad():
                    out_supervised_teacher = meanteacher.teacher(labeled_images) # '$\tilde{p}$'
                
                p_tilde_labeled_soft = torch.softmax(out_supervised_teacher, dim=1)

                # 2. Perform Supervised Quadripartition ($\tilde{p}$ vs. y*) 
                with torch.no_grad():
                    y_tilde_l = torch.argmax(p_tilde_labeled_soft, dim=1)
                    max_prob_tilde_l, max_class_tilde_l = p_tilde_labeled_soft.max(dim=1)
                    # Build threshold map using the SAME global thresholds from the unsupervised part
                    threshold_map_l = torch.zeros_like(max_prob_tilde_l)
                    for c in range(4):
                        threshold_map_l[max_class_tilde_l == c] = class_thresholds_g[c].item()
                    # Define Confidence (teacher vs. threshold) and Consistency (teacher vs. GT)
                    is_confident_l = (max_prob_tilde_l > threshold_map_l)
                    is_unanimous_l = (y_tilde_l == labeled_targets) # $\tilde{y}$ vs. y*
                    
                    # Create the 4 region masks
                    mask_UC_l = (is_unanimous_l & is_confident_l).float()
                    mask_US_l = (is_unanimous_l & ~is_confident_l).float()
                    mask_DC_l = (~is_unanimous_l & is_confident_l).float()
                    mask_DS_l = (~is_unanimous_l & ~is_confident_l).float()

                # 3. Get Supervised Weights (w_UC > w_US > w_DS > w_DC) [cite: 239, 484]
                beta = 3.0
                delta_l = 0.3 # Optimal value for labeled loss [cite: 484]
                
                w_UC_l = generalized_gaussian(torch.tensor(0.0 * delta_l, device="cuda"), beta)
                w_US_l = generalized_gaussian(torch.tensor(1.0 * delta_l, device="cuda"), beta)
                w_DS_l = generalized_gaussian(torch.tensor(2.0 * delta_l, device="cuda"), beta) # DS is 2nd
                w_DC_l = generalized_gaussian(torch.tensor(3.0 * delta_l, device="cuda"), beta) # DC is 3rd

                # Build the final pixel-wise weight map
                weight_map_l = (mask_UC_l * w_UC_l) + \
                            (mask_US_l * w_US_l) + \
                            (mask_DS_l * w_DS_l) + \
                            (mask_DC_l * w_DC_l)

                # 4. Calculate the two-part supervised loss (L_ce^lqh + L_dice^lqh) [cite: 278]
                
                # --- Weighted CE Loss (Eq. 7) ---
                # Get per-pixel CE loss (student vs. GT)
                loss_ce_labeled_unweighted = criterion_ce(out_supervised_student, labeled_targets)
                L_ce_lqh = (loss_ce_labeled_unweighted * weight_map_l).mean()
                
                # --- Weighted Dice Loss (Eq. 8) ---
                p_supervised_soft = torch.softmax(out_supervised_student, dim=1)
                y_true_onehot = F.one_hot(labeled_targets, num_classes=4).permute(0, 3, 1, 2).float()
                
                L_dice_lqh = calculate_weighted_dice_loss(
                    p_supervised_soft, 
                    y_true_onehot, 
                    weight_map_l, 
                    num_classes=4
                )
                
                # --- Final Supervised Loss ---
                supervised_loss = L_ce_lqh + L_dice_lqh

            # Delete labeled tensors immediately (free ~half the input memory)
            del labeled_images, labeled_targets, out_supervised_student,out_supervised_teacher, lab_4, lab_5, lab_6, target_4, target_5, target_6
            torch.cuda.empty_cache()  # Force release

            # Now move/process unlabeled
            unlab_images = unlab_images.cuda(non_blocking=True)
            unlab_images_weak, unlab_images_noisy = apply_unlabeled_augs_batch(
                unlab_images, 
                kornia_weak=dataset_unlabeled.kornia_weak, 
                kornia_strong=dataset_unlabeled.kornia_strong,
                min_weak_ops=0, max_weak_ops=dataset_unlabeled.nr_weak_transforms,
                min_strong_ops=1, max_strong_ops=dataset_unlabeled.nr_strong_transforms
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Unsupervised forwards/losses
                out_unsupervised_student = meanteacher.student(unlab_images_noisy)
                with torch.no_grad():  # Teacher doesn't need grads—saves activation memory
                    out_unsupervised_teacher = meanteacher.teacher(unlab_images_weak)


                #### START METASSL
                p_unlabeled_soft = torch.softmax(out_unsupervised_student, dim=1)
                p_tilde_unlabeled_soft = torch.softmax(out_unsupervised_teacher, dim=1)
                with torch.no_grad():
                    # Get teacher's hard predictions (pseudo-labels)
                    # y_tilde = torch.argmax(p_tilde_unlabeled_soft, dim=1)
                    
                    # Paper Eq. 9: Calculate mean probability for each class [cite: 256]
                    for c in range(4): # Loop 0, 1, 2, 3
                        # Mask for pixels predicted as class 'c'
                        class_mask = (torch.argmax(p_tilde_unlabeled_soft, dim=1) == c)
                        
                        if class_mask.sum() > 0:
                            # Get the probabilities *for class c* where the mask is true
                            mean_prob_c = p_tilde_unlabeled_soft[:, c, :, :][class_mask].mean()
                            
                            # Paper Eq. 10: Update global threshold via EMA [cite: 260]
                            class_thresholds_g[c] = (threshold_ema_alpha * mean_prob_c) + \
                                                    ((1 - threshold_ema_alpha) * class_thresholds_g[c])
                            
                
                with torch.no_grad():
                    # 1. Get hard predictions for both student and teacher
                    y_p = torch.argmax(p_unlabeled_soft, dim=1)
                    y_tilde = torch.argmax(p_tilde_unlabeled_soft, dim=1) # Reference
                    
                    # 2. Get confidence of the reference (teacher)
                    max_prob_tilde, max_class_tilde = p_tilde_unlabeled_soft.max(dim=1)
                    
                    # 3. Get the adaptive threshold for each pixel based on its predicted class
                    # This is tricky to vectorize, but a loop or torch.gather is possible.
                    # For simplicity (and speed), we can build a threshold map:
                    threshold_map = torch.zeros_like(max_prob_tilde)
                    for c in range(4):
                        threshold_map[max_class_tilde == c] = class_thresholds_g[c].item()
                        
                    # 4. Define Confidence and Consistency
                    is_confident = (max_prob_tilde > threshold_map)
                    is_unanimous = (y_p == y_tilde)
                    
                    # 5. Create the 4 region masks 
                    mask_UC = (is_unanimous & is_confident).float()
                    mask_US = (is_unanimous & ~is_confident).float()
                    mask_DC = (~is_unanimous & is_confident).float()
                    mask_DS = (~is_unanimous & ~is_confident).float()

                beta = 3.0
                delta_u = 0.6
                w_UC = generalized_gaussian(torch.tensor(0.0 * delta_u, device="cuda"), beta) # = 1.0
                w_US = generalized_gaussian(torch.tensor(1.0 * delta_u, device="cuda"), beta)
                w_DC = generalized_gaussian(torch.tensor(2.0 * delta_u, device="cuda"), beta)
                w_DS = generalized_gaussian(torch.tensor(3.0 * delta_u, device="cuda"), beta)

                # Get the per-pixel CE loss
                loss_ce_unsupervised = criterion_ce(out_unsupervised_student, y_tilde)

                # Build the final pixel-wise weight map
                weight_map = (mask_UC * w_UC) + \
                            (mask_US * w_US) + \
                            (mask_DC * w_DC) + \
                            (mask_DS * w_DS)

                # Calculate the final heterogeneous loss (Eq. 5) [cite: 215]
                # We take the weighted mean
                L_uqh = (loss_ce_unsupervised * weight_map).mean()

                # Get your consistency ramp-up weight
                current_consistency_weight = get_current_consistency_weight(current_iteration=iter_num,rampup_length=rampup_iterations,max_consistency=max_consistency)

                # THIS REPLACES YOUR OLD `unsupervised_loss`
                unsupervised_loss = current_consistency_weight * L_uqh

                # Your total loss is now the standard supervised + MetaSSL unsupervised
                total_loss = supervised_loss + unsupervised_loss


                ### END METASSL

                # Add uncertainties to dict
                # entropy_per_patch = -torch.sum(pseudo_label * torch.log(pseudo_label + 1e-8), dim=1)


            # CHANGE uncomment when needed again
            # mean_entropies = entropy_per_patch.mean(dim=[1,2])  # If entropy is [B, H, W], mean per patch
            # for wsi_id, mean_ent in zip(unlab_wsi_ids, mean_entropies.cpu()):  # .cpu() only if needed
            #     unc_per_wsi[wsi_id].append(mean_ent.item())
            
            total_loss.backward()
            optimizer.step()
            meanteacher.update_teacher()
            lr_scheduler.step()

            writer.add_scalars('loss/supervised',{'dice':L_dice_lqh.item(),'ce':L_ce_lqh.item()}, iter_num)
            writer.add_scalar('loss/unsupervised',unsupervised_loss.item(), iter_num)
            writer.add_scalars('loss/train', {'supervised':supervised_loss.item(),'unsupervised':unsupervised_loss.item()}, iter_num)


            # Logging Learning Rate
            writer.add_scalar("params/learning_rate", optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('params/consistency_weight', get_current_consistency_weight(current_iteration=iter_num, rampup_length=rampup_iterations, max_consistency=max_consistency), iter_num)
            epoch_train_losses.append(total_loss.item())

            if iter_num % 250 == 0:
                writer.add_scalars('loss/total', {'train':np.mean(epoch_train_losses)}, iter_num)
                # CHANGE this to the nnFilter approach, i.e. after warmup period, drop the 5 WSI ids with the lowest uncertainty, i.e. the model already kind of knows how to classify them
                # unc_per_wsi.clear()
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
                        mean_dice = torch.mean(per_class_dice[1:]).item() # Ignore background
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
    
    with open('./src/config/unlabeled_split_10_11.json') as f:
        unlabeled_wsi_config = json.load(f)
    
    unlabeled_wsi_ids = []
    for unlabeled_group in unlabeled_wsi_config["train"].values():
        unlabeled_wsi_ids.extend(unlabeled_group)

    unlabeled_wsi_ids = list(set(unlabeled_wsi_ids) - set(labeled_wsi) - set(test_wsi))

    logger.info(f"Total Number of labeled WSI {len(labeled_wsi)}\n Total number of unlabeled WSI {len(unlabeled_wsi_ids)}")

    torch.backends.cudnn.benchmark = True # Can be changed to false
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(clean_path)
    train_UPC(args,labeled_wsi=labeled_wsi,val_wsi=test_wsi,unlabeled_wsi=unlabeled_wsi_ids,logs_dir=experiment_Logs_dir)

