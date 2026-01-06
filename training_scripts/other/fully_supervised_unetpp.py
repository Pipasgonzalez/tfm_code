from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from monai.losses.dice import DiceCELoss
from torch.optim import Adam
import torch
import numpy as np
import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Easiest
# cd ~/master_thesis/code
# python -m src.training_scripts.fully_supervised_unetpp
from ..utils.data import LMDataset, create_balanced_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--nclasses", type=int, default=4)
parser.add_argument("--nsamples", type=int, default=100)
parser.add_argument("--lmdb_path", type=str, default="D:/tfm_data/preprocessed/lmdb")

# 1. Cosine Annealing with Warm Restarts (Recommended)
# Why?
# Periodically restarts learning rate, helping escape local minima.
# Works well with limited data and adaptive optimizers like AdamW.
# When to Use:
# Default for most medical segmentation tasks.
# Especially useful for U-Net, U-Net++, or Transformer-based architectures.


def get_model(nclasses: int, device):
    model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=3,
        out_channels=nclasses,
        features=(16, 32, 48, 64, 128, 32),
        deep_supervision=False,
    )
    return model.to(device=device)


def main(
    nclasses: int,
    epochs: int,
    nsamples: int,
    lmdb_path: str = "D:/tfm_data/preprocessed/lmdb",
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(nclasses=nclasses, device=device)
    optim = Adam(params=model.parameters(), lr=1e-3)
    # criterion = DiceCELoss(include_background=True)
    criterion = torch.nn.CrossEntropyLoss()
    torch.nn.loss

    # Dataset, use default settings for most
    ds = LMDataset(lmdb_path=lmdb_path)

    # Dataloder
    dl = create_balanced_dataloader(dataset=ds, total_images=nsamples, batch_size=4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10)
    # Actual Training
    NUM_EPOCHS = epochs

    tot_train_losses = []
    for epoch in range(NUM_EPOCHS):
        train_losses = []
        print(f"Starting Epoch #{epoch+1}")
        model.train()
        for batch in dl:
            patches, targets, _ = batch
            patches = patches.to(device)
            targets = targets.to(device)
            y_hat = model(patches)
            if isinstance(y_hat, list):
                y_hat = y_hat[0]
            targets_ce = targets.squeeze(1)
            loss = criterion(y_hat, targets_ce)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_losses.append(loss.item())
            print(f"Current Batch Loss: {train_losses[-1]}")
        scheduler.step()
        tot_train_losses.append(np.mean(train_losses))


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        nclasses=args.nclasses,
        epochs=args.epochs,
        lmdb_path=args.lmdb_path,
        nsamples=args.nsamples,
    )
