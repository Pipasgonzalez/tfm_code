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
parser.add_argument("--consistency_rampup", type = float, default = 75.0)
parser.add_argument('--bs',type=int,default=6)
parser.add_argument("--max_epochs", type = int, default = 75)
parser.add_argument("--consistency", type = float, default = 1.0)
parser.add_argument('--hpc', action='store_true')
parser.add_argument("--split",type=int,default=1)
parser.add_argument("--seed",type=float,default=47)
parser.add_argument("--workers",type=int,default=0)
parser.add_argument("--patience", type = int, default = 10)
parser.add_argument("--output_dir", type = str, default="./src")
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

def train_UPC(args, labeled_wsi, val_wsi, unlabeled_wsi, logs_dir):
    unc_per_wsi = defaultdict(list)
    # tensorboard_path = os.path.join(logs_dir, "pretrain")
    writer = SummaryWriter(logs_dir)
    model_path = os.path.join(logs_dir,"best_upc.pth")
    student = swin_upc()
    meanteacher = MeanTeacher(student_model=student)
    meanteacher.student.cuda()
    meanteacher.teacher.cuda()

    optimizer = torch.optim.AdamW(params = meanteacher.student.parameters(),lr = 0.008, weight_decay=0.0001)
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

        dataset_unlabeled = LMDBTorchDataset("./dataset_unlabeled_lvl5", ds_type='train',allowed_wsi_ids=unlabeled_wsi,unlabeled=True,return_wsi_id=True) # CHANGE returning whole slide images now
        unlabeled_sampler = torch.utils.data.RandomSampler(dataset_unlabeled,replacement=True,num_samples=len(dataset_labeled_4))
        train_dl_unlab = torch.utils.data.DataLoader(dataset_unlabeled,batch_size=int(args.bs/2), num_workers= args.workers if args.hpc else 0,pin_memory=True,drop_last=True,persistent_workers=True,sampler=unlabeled_sampler)

        dice_eval = DiceScore(num_classes=4, include_background=True, average='none', input_format='index')
        test_dataset_4 = LMDBTorchDataset(lmdb_path="./dataset_lvl4", allowed_wsi_ids=val_wsi, ds_type='test')
        test_dataset_5 = LMDBTorchDataset(lmdb_path="./dataset_lvl5", allowed_wsi_ids=val_wsi, ds_type='test')
        test_dataset_6 = LMDBTorchDataset(lmdb_path="./dataset_lvl6", allowed_wsi_ids=val_wsi, ds_type='test')
        test_loader_4 = torch.utils.data.DataLoader(dataset=test_dataset_4, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        test_loader_5 = torch.utils.data.DataLoader(dataset=test_dataset_5, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        test_loader_6 = torch.utils.data.DataLoader(dataset=test_dataset_6, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)


    batches_per_epoch = min(len(train_dl_lab_4), len(train_dl_unlab)) # Should be the same but just in case
    max_iterations = batches_per_epoch * args.max_epochs
    rampup_iterations = int(max_iterations * 0.5)
    args.consistency_rampup = rampup_iterations

    iter_num = 0
    best_performance = 0
    patience_counter = 0

    cudamove_times = []
    aug_times = []
    train_times = []
    gradient_times = []
    uncertainty_times = []
    for epoch in range(args.max_epochs):
        meanteacher.student.train()
        meanteacher.teacher.eval()

        # IMPORTANT: Create Train Dataloader Each Time. If not it runs out
        train_labeled_loader = zip(train_dl_lab_4, train_dl_lab_5, train_dl_lab_6)
        for (((lab_4, target_4), (lab_5, target_5), (lab_6, target_6)),(unlab_images, unlab_wsi_ids)) in tqdm(zip(train_labeled_loader,train_dl_unlab),desc=f'Training Epoch {epoch+1}', total=min(len(train_dl_lab_4),len(train_dl_unlab))):
            
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear at start of batch
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

            # Now move/process unlabeled
            unlab_images = unlab_images.cuda(non_blocking=True)
            unlab_images_weak, unlab_images_noisy = apply_unlabeled_augs_batch(
                unlab_images, 
                kornia_weak=dataset_unlabeled.kornia_weak, 
                kornia_strong=dataset_unlabeled.kornia_strong[:1] + dataset_unlabeled.kornia_strong[2:], # CHANGE removed elastic deformation from unlabeled augmentations. To be tested lateron
                min_weak_ops=0, max_weak_ops=dataset_unlabeled.nr_weak_transforms,
                min_strong_ops=1, max_strong_ops=dataset_unlabeled.nr_strong_transforms
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Unsupervised forwards/losses
                out_unsupervised_student = meanteacher.student(unlab_images_noisy)
                with torch.no_grad():  # Teacher doesn't need grads—saves activation memory
                    out_unsupervised_teacher = meanteacher.teacher(unlab_images_weak)
                pseudo_label = sm(out_unsupervised_teacher)
                targets_teacher = torch.argmax(pseudo_label, dim=1)

                loss_ce = criterion_ce(out_unsupervised_student,targets_teacher)
                loss_kl = torch.sum(kl_distance(logsm(out_unsupervised_student),sm(out_unsupervised_teacher)),dim=1)
                exp_loss_kl = torch.exp(-loss_kl)
                loss_rect = torch.mean(loss_ce*exp_loss_kl) + torch.mean(loss_kl)

                current_consistency_weight = get_current_consistency_weight(current_iteration=iter_num)
                unsupervised_loss = current_consistency_weight * loss_rect
                total_loss = supervised_loss + unsupervised_loss

                # Add uncertainties to dict
                entropy_per_patch = -torch.sum(pseudo_label * torch.log(pseudo_label + 1e-8), dim=1)

            end_train = time.time()


            mean_entropies = entropy_per_patch.mean(dim=[1,2])  # If entropy is [B, H, W], mean per patch
            for wsi_id, mean_ent in zip(unlab_wsi_ids, mean_entropies.cpu()):  # .cpu() only if needed
                unc_per_wsi[wsi_id].append(mean_ent.item())
            
            end_uncertainty = time.time()
            total_loss.backward()
            optimizer.step()
            meanteacher.update_teacher()

            end_gradient_update = time.time()

            # if iter_num % 50 == 0:
            #     logger.info(f"Average Cuda Move Time: {np.mean(cudamove_times)}\n Average Augmentation Time: {np.mean(aug_times)}\n Average Train Times: {np.mean(train_times)}\n Average Uncertainty Times: {np.mean(uncertainty_times)}\n Average Gradient Times: {np.mean(gradient_times)}")
                
            #     # Clear List Cache
            #     cudamove_times.clear()
            #     aug_times.clear()
            #     train_times.clear()
            #     uncertainty_times.clear()
            #     gradient_times.clear()

            writer.add_scalars('loss/supervised',{'dice':dice_loss_supervised.item(),'focal':focal_loss_supervised.item()}, iter_num)
            writer.add_scalar('loss/unsupervised',unsupervised_loss.item(), iter_num)
            writer.add_scalars('loss/train', {'supervised':supervised_loss.item(),'unsupervised':unsupervised_loss.item()}, iter_num)
            writer.add_scalar('loss/consistency_weight', get_current_consistency_weight(current_iteration=iter_num), iter_num)
            epoch_train_losses.append(total_loss.item())
        
        if iter_num >= max_iterations:
            break # BREAK out of the outer training loop aswell

        writer.add_scalars('loss/total', {'train':np.mean(epoch_train_losses)}, epoch+1)

        # CHANGE this to the nnFilter approach, i.e. after warmup period, drop the 5 WSI ids with the lowest uncertainty, i.e. the model already kind of knows how to classify them
        unc_per_wsi.clear()

        ##### VALIDATION LOOP
        patches_to_log_per_loader = 4  # Log 4 random images per loader (total up to 12)
        val_losses_per_resolution = {}
        loaders = [
            (test_loader_4, "res4"),
            (test_loader_5, "res5"),
            (test_loader_6, "res6")
        ]
        with torch.no_grad():
            dice_eval.reset()
            dice_score_per_resolution = {}

            for loader, res_tag in loaders:
                val_loss_values = []
                # New Dice Score for each resolution
                dice_eval.reset()
                num_batches = len(loader)
                if num_batches < patches_to_log_per_loader:
                    log_indices_set = set(range(num_batches))
                else:
                    log_indices = np.random.choice(num_batches, patches_to_log_per_loader, replace=False)
                    log_indices_set = set(log_indices)

                progress_bar_val = tqdm(loader, 
                                        desc=f"Running Validation on Epoch: {epoch+1} ({res_tag})", 
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

                        # 1. Normalize original image (NumPy)
                        img_normalized = normalize_patch_np(image_patch_np)

                        # 2. Color-map ground truth (NumPy)
                        target_color = map_seg_to_color_np(target_patch_np, COLOR_MAP_NP)

                        # 3. Color-map prediction (NumPy)
                        pred_color = map_seg_to_color_np(pred_patch_np, COLOR_MAP_NP)

                        # 4. Log to TensorBoard
                        tag_prefix = f"val/res{res_tag}_batch{val_idx}"
                        writer.add_image(f"{tag_prefix}_Image", img_normalized, epoch + 1, dataformats='CHW')
                        writer.add_image(f"{tag_prefix}_GroundTruth", target_color, epoch + 1, dataformats='CHW')
                        writer.add_image(f"{tag_prefix}_Prediction", pred_color, epoch + 1, dataformats='CHW')

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
                writer.add_scalars(f"val/dsc_{res_tag}", dice_dict, epoch + 1)
                val_losses_per_resolution[res_tag] = np.mean(val_loss_values)

                # Gets added on top of the Training Loss

            global_loss = np.mean([np.mean(val_losses) for val_losses in val_losses_per_resolution.values()])
            writer.add_scalars("loss/total", {"val":global_loss}, epoch + 1)

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
            break
        # Just to be sure
        meanteacher.student.train()
            
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

