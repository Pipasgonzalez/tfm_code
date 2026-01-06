
import json
import torch
from src.utils.dataset_unlabeled import LMDBTorchDatasetUnlabeled
from src.networks.swin_unet import swin_upc
from monai.losses.dice import DiceLoss
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--experiment_name", type = str, default = "20251211_113428_split1_10001iterations_16bs")
parser.add_argument("--output_dir", type = str, default = "./outputs")

def main(args):
    model = swin_upc()
    # model.load_state_dict(torch.load(f"D:/tfm_data/hpc_results/{MODEL_NAME}/best_supervised.pth"))
    model.load_state_dict(torch.load(f"/home/njagergallego/results/logs/{args.experiment_name}/best_supervised.pth", weights_only=True))
    model.half()
    model.eval()

    # Here we used the old split still
    # split_config = json.load(open("./src/config/splits_3110.json"))
    # selected_splits = split_config.get(f"split_{SPLIT}")
    # labeled_wsi = selected_splits["test_wsi_ids"]
    dataset = LMDBTorchDatasetUnlabeled("./lvl4_unlabeled_normalized", return_wsi_id=False,return_idx=True) # allowed_wsi_ids=labeled_wsi
    loader = torch.utils.data.DataLoader(dataset, batch_size = 32)
    loss_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    sm = torch.nn.Softmax(dim=1)
    diceloss = DiceLoss(include_background=True, to_onehot_y=True,softmax=True, reduction="none")


    # Test if the writing process works:
    np.savez(f"{args.output_dir}/hardness_estimates.npz", idx = np.array([1,2,3]))
    all_keys = []
    all_loss_values = []
    EPS = 1e-10
    with torch.no_grad():
        for patch, keys in tqdm(loader):
            patch = patch.cuda().half()
            logits = model(patch)
            logits = logits.float() # for numerical stability
            probs = torch.softmax(logits, dim=1)
            
            # 2. Calculate Log Probabilities (log(p))
            # We look at C (dim 1)
            log_probs = torch.log(probs + EPS)
            
            # 3. Calculate p * log(p), aja Shannon Entropy
            p_log_p = probs * log_probs
            
            # 4. Sum over classes and negate to get Entropy
            # Formula: - Sum(p * log(p))
            # Result Shape: [B, H, W] (Entropy map per pixel)
            pixel_entropy = -torch.sum(p_log_p, dim=1)
            
            # 5. Average over the spatial dimensions to get one score per patch
            # Result Shape: [B]
            avg_patch_entropy = pixel_entropy.mean(dim=(1, 2))

            all_keys.extend(list(keys))
            all_loss_values.extend(avg_patch_entropy.float().cpu().tolist())


    np.savez(f"{args.output_dir}/hardness_estimates.npz", idx = all_keys, hardness = all_loss_values)
    np.savez("/home/njagergallego/src/config/hardness_unlabeled.npz", idx = all_keys, hardness = all_loss_values)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)