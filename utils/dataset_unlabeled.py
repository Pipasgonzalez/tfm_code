import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, Subset
from pathlib import Path
from PIL import Image
import pickle
import lmdb
from tqdm import tqdm
import imageio
import numpy as np
import os
from pathlib import Path
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F
from torchvision import tv_tensors
from typing import Literal
from imageio.v3 import imread
from monai.transforms.spatial.array import Rand2DElastic
from monai.transforms.spatial.dictionary import Rand2DElasticd
import kornia.augmentation as K
from torch.utils.data import Sampler
import scipy
import io
import zlib
from collections import Counter
import random
from loguru import logger
import matplotlib.pyplot as plt

# Imports all augmentations from the Nijmegen Implementation of RandAugment script
import src.utils.randaugment as aug_utils

class LMDBTorchDatasetUnlabeled(Dataset):
    def __init__(self, lmdb_path: str,allowed_wsi_ids: list[str] | None = None,augmentation_type = 'v3', return_wsi_id = False, included_sampling_categories = [0,1], return_idx = False):
        self.lmdb_path = lmdb_path
        self.env = None
        self.return_idx = return_idx
        self.augmentation_type = augmentation_type
        self.return_wsi_id = return_wsi_id
        self.included_sampling_categories = included_sampling_categories
        # --- 1. Load ALL data and map to a single dictionary structure ---
        data_by_key = {}
        temp_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.metadata = []
        with temp_env.begin() as txn:
            # Load metadata (keys and WSI IDs)
            metadata_buffer = txn.get(b"__metadata")
            metadata_array = np.load(io.BytesIO(metadata_buffer), allow_pickle=True)
            # if "coords_x" not in metadata_array:
            #     logger.info("Metadata Array not complete, using npz")
            #     metadata_array = np.load("./src/config/metadata_lvl4_unlabeled.npz", allow_pickle=True)
            #     print(len(metadata_array["keys"]))

            for key, wsi_id, coord_x, coord_y in zip(metadata_array["keys"], metadata_array["wsi_ids"], metadata_array["coords_x"], metadata_array["coords_y"]):
                key_str = key.decode("utf-8")
                data_by_key[key_str] = {"wsi_id": wsi_id, "coord_x":coord_x, "coord_y":coord_y} #"unlab_category": unlabeled_lookup.get(key_str, None)} # Here I want to add the unlab_category

                
        final_keys = []
        allowed_wsi_set = set(allowed_wsi_ids) if allowed_wsi_ids else None
        unique_wsi_ids = []
        # Iterate over all data loaded from LMDB
        for key, data in data_by_key.items():
            
            # Filter 1: WSI ID Check
            if allowed_wsi_set and data["wsi_id"] not in allowed_wsi_set:
                if data["wsi_id"] not in unique_wsi_ids:
                    unique_wsi_ids.append(data["wsi_id"])
                continue

            # if data["unlab_category"] not in self.included_sampling_categories:
            #     continue
                
                # If all checks pass, keep the key
            final_keys.append(key)
        
        # Sorting the keys guarantees self.metadata and self.labeled_categories are in the same order
        final_keys.sort() 

        self.metadata = []
        self.composition_keys = []
        for key in final_keys:
            data = data_by_key[key]
            
            # Build ALIGNED self.metadata (needed for __getitem__ index i)
            self.metadata.append((key, data["wsi_id"], data["coord_x"], data["coord_y"]))
            comp_key = f"{data['wsi_id']}_{data['coord_x']}_{data['coord_y']}"
            self.composition_keys.append(comp_key)

        self.length = len(self.metadata)
        self.indices_per_key = {}
        for idx, (_, wsi_id, _, _) in enumerate(self.metadata):
             # The WSI ID is the key for grouping.
             # The index (idx) is the actual index used by the PyTorch Dataloader.
             if wsi_id not in self.indices_per_key:
                  self.indices_per_key[wsi_id] = []
             self.indices_per_key[wsi_id].append(idx)

            
        # KORNIA AUGMENTATIONS
        self.imgtotensor = v2.PILToTensor()
        self.nr_weak_transforms = 2
        self.nr_strong_transforms = 1
        self.nr_labeled_transforms = 4
        self.kornia_weak = [K.RandomHorizontalFlip(p=1), K.RandomVerticalFlip(p=1), K.RandomRotation90(times=(-3,3), p=1)]
        self.kornia_strong = [K.ColorJiggle(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.9,1.1), hue=(-0.05,0.05), p=1), K.RandomGaussianNoise(mean=0, std=0.1, p=1), K.RandomGaussianBlur(kernel_size=(7,7), sigma=(0.1,1), p=1)]
        # self.kornia_strong = [K.RandomGaussianNoise(mean=0, std=0.1, p=1)]
        self.kornia_all = self.kornia_weak + self.kornia_strong

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Happens once per Dataloader Worker

        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        
        key, wsi_id, coord_x, coord_y = self.metadata[idx]
        with self.env.begin() as txn:
            patch_data = txn.get(key.encode() + b"_patch")
            patch = Image.open(io.BytesIO(patch_data))

        patch = self.imgtotensor(patch).float() / 255.0
        if self.return_wsi_id:
            return patch, wsi_id
        elif self.return_idx:
            composite_key = "_".join([str(wsi_id), str(coord_x), str(coord_y)])
            return patch, composite_key
        else:
            return patch


    def get_labeled_category_counts(self):
        categories = np.array([cat for _, cat in self.labeled_categories])
        return np.bincount(categories, minlength=2) # THIS WAS IT
    

    def get_wsi_ids(self):
        # Don't have to convert strings to int. Gets handled by CV
        return [wsi_id for _, wsi_id in self.metadata]

    def __del__(self):
        if self.env:
            self.env.close()


class HardnessBatchSampler(Sampler):
    """
    Samples indices until a batch of batch_size is filled with patches 
    whose parent WSI ID meets the entropy threshold requirement.
    """
    def __init__(self, dataset:LMDBTorchDatasetUnlabeled, batch_size, threshold_hardness, hardness_per_key,  max_attempts=10000):
        self.dataset:LMDBTorchDatasetUnlabeled = dataset
        self.batch_size = batch_size
        self.threshold_hardness = threshold_hardness
        self.hardness_per_key = hardness_per_key
        self.max_attempts = max_attempts
        
        self.num_samples = len(dataset)
        # Determine the number of batches to simulate one epoch
        self.num_batches = len(self.dataset) // self.batch_size 

        #self.keys =#list(self.dataset.keys) if hasattr(self.dataset, 'keys') else [str(i) for i in range(len(dataset))]
        if hasattr(self.dataset, 'composition_keys'):
            self.keys = self.dataset.composition_keys
        else:
            raise AttributeError("Dataset must have 'composite_keys' attribute for HardnessBatchSampler")
        
        hardness_list = []
        missing_keys = []
        for k in self.keys:
            if k in self.hardness_per_key:
                hardness_list.append(self.hardness_per_key[k])
            else:
                missing_keys.append(k)
        
        # 3. CRITICAL: Crash if alignment is off
        if missing_keys:
            raise KeyError(f"Sampler Alignment Error: {len(missing_keys)} keys found in the dataset are MISSING from the hardness .npz file.\n"
                           f"First 5 missing keys: {missing_keys[:5]}")
        
        self.hardness_values = np.clip(np.array(hardness_list), a_min=0, a_max=1.0)

    def __iter__(self):
        valid_indices = np.where(self.hardness_values >= self.threshold_hardness)[0]
        np.random.shuffle(valid_indices)
        current_idx = 0
        for _ in range(self.num_batches):
            if current_idx + self.batch_size > len(valid_indices):
                np.random.shuffle(valid_indices)
                current_idx = 0
            batch = valid_indices[current_idx : current_idx + self.batch_size]
            current_idx += self.batch_size
            yield batch.tolist()

    def __len__(self):
        return self.num_batches
    
class FastHardnessBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, threshold_hardness, hardness_per_key):
        self.dataset = dataset
        self.batch_size = batch_size
        self.threshold_hardness = threshold_hardness
        self.hardness_per_key = hardness_per_key
        
        # Determine number of batches (for len())
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // self.batch_size 

        # --- 1. Alignment Logic (Same as before) ---
        if hasattr(self.dataset, 'composition_keys'):
            self.keys = self.dataset.composition_keys
        else:
            raise AttributeError("Dataset must have 'composition_keys'.")
        
        hardness_list = []
        missing_keys = []
        for k in self.keys:
            if k in self.hardness_per_key:
                hardness_list.append(self.hardness_per_key[k])
            else:
                missing_keys.append(k)
        
        if missing_keys:
            raise KeyError(f"Sampler Alignment Error: {len(missing_keys)} keys missing.")
        
        # --- 2. THE OPTIMIZATION: Pre-Sort by Hardness ---
        # We convert to numpy arrays once
        raw_hardness = np.clip(np.array(hardness_list), 0.0, 1.0)
        
        # Argsort gives us the indices that would sort the array
        # This takes O(N log N) but only runs ONCE during init
        self.sorted_args = np.argsort(raw_hardness)
        
        # We store the hardness values in sorted order
        self.sorted_hardness = raw_hardness[self.sorted_args]
        
        # We store the map: sorted_position -> original_dataset_index
        self.sorted_indices = self.sorted_args 

    def __iter__(self):
        # This loop runs every time the DataLoader needs more batches
        for _ in range(self.num_batches):
            # 1. Find the cut-off point efficiently (Binary Search)
            # This is instantaneous even for 1M+ images
            start_idx = np.searchsorted(self.sorted_hardness, self.threshold_hardness)            
            # Safety: If threshold is too high (pool empty), fallback to taking the hardest available chunk
            # or reset to 0. Here we default to "all data" to prevent crash, 
            # but you could also set it to len-batch_size to force hardest batches.
            if start_idx >= self.num_samples:
                start_idx = 0 
                
            # 2. Vectorized Sampling
            # We want 'batch_size' random integers between [start_idx, end_of_array]
            # This effectively samples only from valid hardness >= threshold
            chosen_offsets = np.random.randint(start_idx, self.num_samples, size=self.batch_size)
            
            # 3. Map back to original dataset indices
            batch_indices = self.sorted_indices[chosen_offsets]
            
            # Yield the list (DataLoader expects a list for BatchSampler)
            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches
    
def get_unlabeled_weighted_hardnesssampler(dataset_unlabeled, pred_hardness_per_key):
    epsilon_weight = 0.1
    if not hasattr(dataset_unlabeled, 'composition_keys'):
            raise AttributeError("Dataset must have 'composition_keys' attribute for alignment.")
    ordered_weights = []
    missing_keys = []
    for k in dataset_unlabeled.composition_keys:
        if k in pred_hardness_per_key:
            # Weight = Hardness + Epsilon
            w = np.clip(pred_hardness_per_key[k], 0.0, 1.0) + epsilon_weight
            ordered_weights.append(w)
        else:
            missing_keys.append(k)
        
        if missing_keys:
             raise KeyError(f"Sampler Alignment Error: {len(missing_keys)} keys found in dataset are MISSING from hardness file.\nFirst 5: {missing_keys[:5]}")

        # Create Standard PyTorch Weighted Sampler
    weights_tensor = torch.DoubleTensor(ordered_weights)
    weighted_sampler = torch.utils.data.WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
    return weighted_sampler

def fit_normal(values, clip_values:list = []):
    if len(clip_values) == 2 and clip_values[0]<clip_values[1]:
        values = values[
            (values >= clip_values[0]) & (values <= clip_values[1])
        ]
    mean,std = scipy.stats.norm.fit(values)
    return mean,std

import numpy as np
from torch.utils.data import Sampler

class HardnessNormalBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, threshold_hardness, mean_hardness, std_hardness, hardness_per_key, max_attempts=10000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.threshold_hardness = threshold_hardness
        self.mean_hardness = mean_hardness
        self.std_hardness = std_hardness
        self.hardness_per_key = hardness_per_key
        self.max_attempts = max_attempts
        self.fixed_righthand_threshhold = 0.8
        self.alpha_normal_params = 0.9999
        self.alpha_hardness_filtering = 0.9999
        self.num_samples = len(dataset)
        self.num_batches = len(self.dataset) // self.batch_size

        # Use composition_keys, as these are the keys that map 1:1 to patch indices
        if hasattr(self.dataset, 'composition_keys'):
            self.keys = self.dataset.composition_keys
        else:
            raise AttributeError("Dataset must have 'composition_keys' attribute for HardnessBatchSampler")
        
        # 1. Align hardness values with the ordered list of all patch keys (self.keys)
        hardness_list = []
        missing_keys = []
        for k in self.keys:
            if k in self.hardness_per_key:
                hardness_list.append(self.hardness_per_key[k])
            else:
                missing_keys.append(k)
        
        if missing_keys:
            print(missing_keys)
            print(self.keys)
            raise KeyError(f"Sampler Alignment Error: {len(missing_keys)} keys found in the dataset are MISSING from the hardness dictionary.")
        
        self.hardness_values = np.clip(np.array(hardness_list), a_min=0, a_max=1.0)
        
        # The list of indices we can sample from (0 to N-1)
        self.all_indices = np.arange(self.num_samples)
        

        raw_pdf = scipy.stats.norm.pdf(self.hardness_values, loc=self.mean_hardness, scale=self.std_hardness)
        artifact_threshold = 0.7
        raw_pdf = np.where(self.hardness_values > artifact_threshold, 0.0, raw_pdf)
        # self.sampling_weights = np.where(
        #     (self.hardness_values >= self.threshold_hardness) & (self.hardness_values <= self.fixed_righthand_threshhold),
        #     raw_pdf,
        #     0.0
        # )
        self.sampling_weights = raw_pdf
        total_weight = self.sampling_weights.sum()
        self.sampling_weights /= total_weight

    def update_sampling_weights(self):
        raw_pdf = scipy.stats.norm.pdf(self.hardness_values, loc=self.mean_hardness, scale=self.std_hardness)
        # self.sampling_weights = np.where(
        #     (self.hardness_values >= self.threshold_hardness) & (self.hardness_values <= self.fixed_righthand_threshhold),
        #     raw_pdf,
        #     0.0
        # )
        artifact_threshold = 0.8
        raw_pdf = np.where(self.hardness_values > artifact_threshold, 0.0, raw_pdf)
        self.sampling_weights = raw_pdf
        total_weight = self.sampling_weights.sum()# NEW
        self.sampling_weights /= total_weight
    
    def update_threshhold(self, average_UC):
        adapted_average_uc = average_UC.item() /1.25 # CHANGE to make the average UC mask be on the same sclae as the threshhold value range (0 to 0.8)
        new_val = (self.alpha_hardness_filtering * self.threshold_hardness) + ((1-self.alpha_hardness_filtering) * adapted_average_uc)
        self.threshold_hardness = float(np.clip(new_val, 0.0, 0.7))

    def update_normal_params(self, training_hardness_values):
        new_mean, new_std = fit_normal(np.array(training_hardness_values))

        if new_mean >= self.mean_hardness:
            new_ema_mean = self.alpha_normal_params*self.mean_hardness - (1-self.alpha_normal_params) * np.clip(new_mean, 0.0,0.8).item()
        else:
            new_ema_mean = self.alpha_normal_params*self.mean_hardness + (1-self.alpha_normal_params) * np.clip(new_mean, 0.0,0.8).item()
        new_ema_std = self.alpha_normal_params*self.std_hardness + (1-self.alpha_normal_params) * new_std
        # if new_ema_mean < self.threshold_hardness:
        #     logger.info(f"Mean Update not possible. New mean smaller than adaptive threshhold:{new_ema_mean}<{self.threshold_hardness} Clipping mean to threshold value")
        #     self.mean_hardness, self.std_hardness = self.threshold_hardness, new_ema_std
        #if new_ema_mean >= 0.2 and new_ema_mean <= 0.6 and new_ema_std <= 0.5:
        # else:
        self.mean_hardness, self.std_hardness = np.clip(new_ema_mean, 0.0, 0.8).item(), new_ema_std
        self.update_sampling_weights()
        # else:
        #     logger.info(f"Could not update normal distribution parameters because mean shifted too far or standard deviation too strong. EMA: {new_ema_mean} STD: {new_ema_std}")

    def update_eval(self,new_mean):
        self.mean_hardness = new_mean
        self.update_sampling_weights()

    def update_normal_params_UC(self, average_UC):
        # scaled_uc = average_UC.item()
        min_hardness_target = 0.0
        max_hardness_target = 0.36 # FINAL
        target_mean = min_hardness_target + (average_UC.item() * (max_hardness_target - min_hardness_target))
        new_mean = (self.alpha_normal_params * self.mean_hardness) + \
                   ((1 - self.alpha_normal_params) * target_mean)
        self.mean_hardness = new_mean
        # self.std_hardness = 0.15 # Or decay it: self.std_hardness * 0.99
        self.update_sampling_weights()

    def __iter__(self):
        
        for _ in range(self.num_batches):
            # np.random.choice samples the indices based on their corresponding weights
            # replace=False ensures we do not select the same patch twice in one batch
            batch_indices = np.random.choice(
                self.all_indices, 
                size=self.batch_size, 
                replace=False, 
                p=self.sampling_weights
            )
                
            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches