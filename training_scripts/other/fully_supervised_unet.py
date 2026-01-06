# cd ~/master_thesis/code
# python -m src.training_scripts.fully_supervised_unet
import torch.nn as nn
import torch
import argparse
from monai.networks.nets.basic_unet import BasicUNet
from monai.losses.dice import DiceCELoss
from src.utils.data import LMDBTorchDataset, create_balanced_dataloader
import numpy as np
from torch.utils.data import Subset, Dataset
import os
import csv
from datetime import datetime
from pathlib import Path
import math
import pytz
from sklearn.model_selection import GroupKFold
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--nclasses", type=int, default=4)
parser.add_argument("--nsamples", type=int, default=50)
parser.add_argument("--n_splits", type=int, default=4)
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument(
    "--lmdb_path", type=str, default="D:/tfm_data/preprocessed/lmdb_raw"
)
parser.add_argument("--outputs_path", type=str, default="./src/logs")

logger.remove(0)
logger.add("./src/logs/test.log")


def perform_training_step(patches, targets, device, optim, model, criterion):
    patches = patches.to(device)
    targets = targets.to(device)

    optim.zero_grad()
    y_hat = model(patches)
    if isinstance(y_hat, list):
        y_hat = y_hat[0]
    loss = criterion(y_hat, targets)
    loss.backward()
    optim.step()
    return loss.item(), y_hat


def perform_validation_step(patches, targets, device, model, criterion):
    patches, targets = patches.to(device), targets.to(device)
    y_hat = model(patches)
    if isinstance(y_hat, list):
        y_hat = y_hat[0]
    loss = criterion(y_hat, targets)
    return loss.item(), y_hat


def get_model(nclasses: int, device):
    model = BasicUNet(
        spatial_dims=2,  # because we have 2D Images
        in_channels=3,  # because RGB
        out_channels=nclasses,  # depends on the number of classes
        features=(
            16,
            16,
            32,
            64,
            128,
            16,
        ),  # this is half of the default values. For less computationally heavy networks, divide eveything by 2
    )
    return model.to(device=device)


def main(
    nclasses: int,
    epochs: int,
    nsamples: int,
    outputs_path: str,
    n_splits: int,
    batch_size: int,
    lmdb_path: str = "D:/tfm_data/preprocessed/lmdb_raw",
):
    torch.cuda.empty_cache()
    amsterdam_tz = pytz.timezone("Europe/Amsterdam")
    start_time = datetime.now(amsterdam_tz)
    filename_time_string = start_time.strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting Training at {filename_time_string}")

    # Check if output path exists
    if not os.path.isdir(outputs_path):
        raise ModuleNotFoundError("The provided output path is not available")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cur_proj_dir = outputs_path + f"/unet_{epochs}epochs_{timestamp}"
    Path(cur_proj_dir).mkdir(exist_ok=True)
    outputs_path = cur_proj_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(nclasses=nclasses, device=device)
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = DiceCELoss(include_background=True, to_onehot_y=True)

    # CSV setup for logging
    csv_path = os.path.join(outputs_path, "training_metrics_supervised.csv")
    fieldnames = [
        "fold",
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    ]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Dataset, use default settings for most
    dataset = LMDBTorchDataset(lmdb_path=lmdb_path)
    # Get indices and groups
    indices = np.arange(len(dataset))
    groups = dataset.get_wsi_ids()  # List of str wsi_id

    # Initialize splitter
    gkf = GroupKFold(n_splits=n_splits)

    best_val_acc = 0.0
    best_model_path = (
        f"./src/pretrained_models/best_unet_supervised_{filename_time_string}.pth"
    )
    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        logger.info(f"Fold {fold + 1}/{n_splits}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create balanced loaders
        train_loader = create_balanced_dataloader(
            train_subset,
            total_images=int(nsamples * 0.8) if nsamples else None,
            batch_size=batch_size,
            replacement=True,
        )
        val_loader = create_balanced_dataloader(
            val_subset,
            total_images=int(nsamples * 0.2) if nsamples else None,
            batch_size=batch_size,
            replacement=False,  # Optional: no replacement for val
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10)

        tot_train_losses = []
        for epoch in range(epochs):
            train_losses, train_correct = [], 0
            total_train = 0
            model.train()
            for batch in train_loader:
                patches, targets = batch
                patches, targets = patches.to(device), targets.to(device)
                loss, y_hat = perform_training_step(
                    patches, targets, device, optim, model, criterion
                )
                train_losses.append(loss)
                # Accuracy calculation
                _, predicted = torch.max(y_hat, 1)
                if targets.dim() == 4:
                    targets = targets.squeeze(1)  # [B, H, W]
                train_correct += (predicted == targets).float().sum().item()
                total_train += targets.numel()

            train_loss = np.mean(train_losses)
            train_acc = train_correct / total_train

            # Validation
            model.eval()
            val_losses, val_correct = [], 0
            total_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    patches, targets = batch
                    patches, targets = patches.to(device), targets.to(device)
                    loss, y_hat = perform_validation_step(
                        patches, targets, device, model, criterion
                    )
                    val_losses.append(loss)

                    _, predicted = torch.max(y_hat, 1)
                    if targets.dim() == 4:
                        targets = targets.squeeze(1)  # [B, H, W]
                    val_correct += (predicted == targets).float().sum().item()
                    total_val += targets.numel()

            val_loss = np.mean(val_losses)
            val_acc = val_correct / total_val
            # Save the best model across all folds
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                logger.info(
                    f"✅ New best model saved at fold {fold + 1}, epoch {epoch + 1} with val_acc={val_acc:.4f}"
                )

            # Log to CSV
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(
                    {
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                    }
                )
            logger.info(
                f"Fold {fold+1} Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            scheduler.step()
            tot_train_losses.append(np.mean(train_losses))


if __name__ == "__main__":
    args = parser.parse_args()
    # python -m src.training_scripts.fully_supervised_unet --epochs 50 --nsamples 1000 --batchsize 32 --n_splits 5
    main(
        nclasses=args.nclasses,
        epochs=args.epochs,
        lmdb_path=args.lmdb_path,
        nsamples=args.nsamples,
        n_splits=args.n_splits,
        batch_size=args.batchsize,
        outputs_path=args.outputs_path,
    )
