import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, Subset, Sampler
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
from loguru import logger
# from src.preproc_scripts.lmdb_dataset import SAMPLING_CATEGORIES
import json
import io
import zlib
from collections import Counter
import random

# Imports all augmentations from the Nijmegen Implementation of RandAugment script
import src.utils.randaugment as aug_utils

class LMDBTorchDataset(Dataset):
    def __init__(self, lmdb_path: str,  ds_type:Literal['train','test'],allowed_wsi_ids: list[str] | None = None, skip_val = False, return_wsi_id = False, included_sampling_categories = [0,1,2], return_idx = False):
        self.ds_type = ds_type
        self.included_sampling_categories = included_sampling_categories
        self.lmdb_path = lmdb_path
        self.env = None
        self.return_wsi_id = return_wsi_id
        self.return_idx = return_idx
        # --- 1. Load ALL data and map to a single dictionary structure ---
        data_by_key = {}
        temp_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        with temp_env.begin() as txn:
            # Load metadata (keys and WSI IDs)
            metadata_buffer = txn.get(b"__metadata")
            metadata_array = np.load(io.BytesIO(metadata_buffer), allow_pickle=True)
            for key, wsi_id, coord_x, coord_y in zip(metadata_array["keys"], metadata_array["wsi_ids"], metadata_array["coords_x"], metadata_array["coords_y"]):
                key_str = key.decode("utf-8")
                data_by_key[key_str] = {"wsi_id": wsi_id, "labeled_cat": None, "unlabeled_cat": None, "coords_x":coord_x, "coords_y":coord_y}

            if not skip_val:
                # Load Labeled Categories
                labled_categories_buffer = txn.get(b"__labeled_categories")
                labeled_cats_array = np.load(io.BytesIO(labled_categories_buffer), allow_pickle=True)
                for key, cat in zip(labeled_cats_array["keys"], labeled_cats_array["category"]):
                    data_by_key[key.decode("utf-8")]["labeled_cat"] = cat
                
        final_keys = []
        allowed_wsi_set = set(allowed_wsi_ids) if allowed_wsi_ids else None
        
        # Iterate over all data loaded from LMDB
        for key, data in data_by_key.items():
            
            # Filter 1: WSI ID Check
            if allowed_wsi_set and data["wsi_id"] not in allowed_wsi_set:
                continue

            if self.ds_type != "test" and int(data['labeled_cat']) not in self.included_sampling_categories:
                continue

            # If all checks pass, keep the key
            final_keys.append(key)
        
        # Sorting the keys guarantees self.metadata and self.labeled_categories are in the same order
        final_keys.sort()

        self.metadata = []
        self.labeled_categories = []
        self.unlabeled_categories = []
        
        for key in final_keys:
            data = data_by_key[key]
            
            # Build ALIGNED self.metadata (needed for __getitem__ index i)
            self.metadata.append((key, data["wsi_id"], data["coords_x"], data["coords_y"]))
            
            # Build ALIGNED self.labeled_categories (needed for WeightedRandomSampler)
            if not skip_val and data["labeled_cat"] is not None:
                self.labeled_categories.append((key, data["labeled_cat"]))
        
        if not skip_val and len(self.metadata) != len(self.labeled_categories):
            raise RuntimeError("CRITICAL ALIGNMENT ERROR: Filtered metadata and labeled categories do not match length.")

        self.length = len(self.metadata)
        self.imgtotensor = v2.PILToTensor()
        self.N_OPS = 3
        self.random_operations = ['randomvertical', 'randomhorizontal', 'randomrotation']
    
    def transform(self, image, target, ds_type):
        image = self.imgtotensor(image) # Output: (C, H, W) in [0, 255]
        image = image.float() / 255.0
        target = torch.from_numpy(target).long()

        if ds_type == 'train':
            n_ops = random.randint(0,self.N_OPS)
            if n_ops == 0:
                return image,target
            else:
                target = target.unsqueeze(0)  # (1, H, W)
                for n in range(n_ops):
                    operation = random.choice(self.random_operations)
                    if operation == 'randomvertical':
                        image = F.vflip(image)
                        target = F.vflip(target)
                    elif operation =='randomhorizontal':
                        image = F.hflip(image)
                        target = F.hflip(target)
                    elif operation =='randomrotation':
                        degree = random.choice([-90,-180,-270,90,180,270])
                        image = F.rotate(image, degree)
                        target = F.rotate(target, degree)
                target = target.squeeze(0)
        return image,target

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Happens once per Dataloader Worker

        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        
        key, wsi_id, _,_ = self.metadata[idx]
        with self.env.begin() as txn:
            patch_data = txn.get(key.encode() + b"_patch")
            patch = Image.open(io.BytesIO(patch_data))
            target_data = txn.get(key.encode() + b"_target")
            target = imread(io.BytesIO(target_data))
        patch,target = self.transform(patch,target,ds_type=self.ds_type)
        if self.ds_type == 'train':
            if self.return_wsi_id:
                return patch, target, wsi_id
            else:
                return patch, target
        else:
            if self.return_idx:
                return patch, target, key
            else:
                return patch, target, wsi_id

    def get_labeled_category_counts(self):
        categories = np.array([cat for _, cat in self.labeled_categories])
        return np.bincount(categories)
    

    def get_wsi_ids(self):
        # Don't have to convert strings to int. Gets handled by CV
        return [wsi_id for _, wsi_id in self.metadata]

    def __del__(self):
        if self.env:
            self.env.close()


def create_balanced_dataloader(
    dataset: LMDBTorchDataset,
    num_workers:int,
    total_images: int = None,
    batch_size=32,
    replacement=True,
    hpc = True
):
    # If it's the original LMDataset object, use the original methods
    category_counts = dataset.get_labeled_category_counts()
    weights = 1.0 / np.maximum(category_counts, 1)
    sample_weights = np.array(
        [weights[cat] for _, cat in dataset.labeled_categories]
    )

    logger.info(f"Unique Values of the weights assigned for the random sampler:{np.unique(sample_weights)}")
    num_samples = total_images if total_images is not None else len(dataset)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=num_samples, replacement=replacement
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if hpc else False
    )
    return dataloader



class FaberSampler(Sampler):
    """
    Balanced sampler:
    - For each batch, sample N patches for each class in self.classes
    - Patches are selected based on classes present (from classes_per_patch.json)
    """

    def __init__(self, dataset, classes_per_patch_path: str = "./src/config/classes_per_patch_lvl4.json",
                 classes=(0,1,2,3), samples_per_class=4):
        self.dataset = dataset
        self.classes = list(classes)
        self.samples_per_class = samples_per_class

        # Load the JSON file
        with open(classes_per_patch_path) as f:
            self.classes_per_patch = json.load(f)

        # Build mapping: composite_key → dataset index
        # Composite key = wsi_id_coordx_coordy
        self.key_to_idx = {}
        for idx, (key, wsi_id, cx, cy) in enumerate(self.dataset.metadata):
            composite = f"{wsi_id}_{cx}_{cy}"
            self.key_to_idx[composite] = idx

        # Build buckets: class → list of dataset indices
        self.class_buckets = {c: [] for c in self.classes}

        for composite_key, present_classes in self.classes_per_patch.items():
            if composite_key in self.key_to_idx:
                idx = self.key_to_idx[composite_key]
                for cls in present_classes:
                    if cls in self.class_buckets: 
                        self.class_buckets[cls].append(idx)

        # Basic length
        self.num_batches = len(self.dataset) // (len(self.classes) * samples_per_class)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                candidates = self.class_buckets[cls]
                chosen = random.sample(candidates, self.samples_per_class)
                batch.extend(chosen)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches