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
# from src.preproc_scripts.lmdb_dataset import SAMPLING_CATEGORIES

import io
import zlib
from collections import Counter
import random

# Imports all augmentations from the Nijmegen Implementation of RandAugment script
import src.utils.randaugment as aug_utils

class LMDBTorchDataset(Dataset):
    def __init__(self, lmdb_path: str,  ds_type:Literal['train','test'],allowed_wsi_ids: list[str] | None = None, unlabeled=False, skip_val = False,augmentation_type = 'v3', return_wsi_id = False):
        self.ds_type = ds_type
        self.lmdb_path = lmdb_path
        self.env = None
        self.augmentation_type = augmentation_type
        self.return_wsi_id = return_wsi_id
        self.unlabeled = unlabeled
        # --- 1. Load ALL data and map to a single dictionary structure ---
        data_by_key = {}
        temp_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        with temp_env.begin() as txn:
            # Load metadata (keys and WSI IDs)
            metadata_buffer = txn.get(b"__metadata")
            metadata_array = np.load(io.BytesIO(metadata_buffer), allow_pickle=True)
            for key, wsi_id in zip(metadata_array["keys"], metadata_array["wsi_ids"]):
                key_str = key.decode("utf-8")
                data_by_key[key_str] = {"wsi_id": wsi_id, "labeled_cat": None, "unlabeled_cat": None}

            if not skip_val:
                # Load Labeled Categories
                if not unlabeled:
                    labled_categories_buffer = txn.get(b"__labeled_categories")
                    labeled_cats_array = np.load(io.BytesIO(labled_categories_buffer), allow_pickle=True)
                    for key, cat in zip(labeled_cats_array["keys"], labeled_cats_array["category"]):
                        data_by_key[key.decode("utf-8")]["labeled_cat"] = cat

                # Load Unlabeled Categories
                if unlabeled:
                    unlabled_categories_buffer = txn.get(b"__unlabeled_categories")
                    unlabeled_cats_array = np.load(io.BytesIO(unlabled_categories_buffer))
                    for key, cat in zip(unlabeled_cats_array["keys"], unlabeled_cats_array["category"]):
                        data_by_key[key.decode("utf-8")]["unlabeled_cat"] = cat
                
        final_keys = []
        allowed_wsi_set = set(allowed_wsi_ids) if allowed_wsi_ids else None
        
        # Iterate over all data loaded from LMDB
        for key, data in data_by_key.items():
            
            # Filter 1: WSI ID Check
            if allowed_wsi_set and data["wsi_id"] not in allowed_wsi_set:
                continue

            if data['labeled_cat'] == 2 and not unlabeled:
                continue

            # For unlabeled train dataset: only include samples where the 'unlabeled' flag is 1
            if unlabeled:
                if data["unlabeled_cat"] != 1:
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
            self.metadata.append((key, data["wsi_id"]))
            
            # Build ALIGNED self.labeled_categories (needed for WeightedRandomSampler)
            if not skip_val and data["labeled_cat"] is not None:
                self.labeled_categories.append((key, data["labeled_cat"]))
            
            # Build ALIGNED self.unlabeled_categories (optional, but good practice)
            if not skip_val and data["unlabeled_cat"] is not None:
                self.unlabeled_categories.append((key, data["unlabeled_cat"]))
        
        # Final sanity check and length
        if not unlabeled:
            if not skip_val and len(self.metadata) != len(self.labeled_categories):
                raise RuntimeError("CRITICAL ALIGNMENT ERROR: Filtered metadata and labeled categories do not match length.")
        else:
            if not skip_val and len(self.metadata) != len(self.unlabeled_categories):
                raise RuntimeError("CRITICAL ALIGNMENT ERROR: Filtered metadata and unlabeled categories do not match length.")

        self.length = len(self.metadata)


        self.hf = v2.RandomHorizontalFlip(p=0.5)
        self.vf = v2.RandomVerticalFlip(p=0.5)
        self.jitter = v2.ColorJitter(
            brightness=(0.65, 1.35),    # Random between 0.7-1.3 (±30%)
            contrast=(0.5, 1.5),      # Random between 0.7-1.3 (±30%)
            saturation=(0.9, 1.1),    # Random between 0.7-1.3 (±30%)
            hue=(-0.1, 0.1)         # Random between -0.02 to 0.02 (reduced from 0.05)
        )
        self.imgtotensor = v2.PILToTensor()
        
        # --- 3. RandAugment Hyperparameters ---
        # These are the 'n' and 'm' values
        # The paper found optimal N=4/5, M=21-25
        self.N_OPS = 4  # (n) Number of operations to apply for labeled dataset
        self.N_OPS_UNLABELED = 2

        self.random_operations = ['randomvertical, randomhorizontal, randomrotation, scaling', 'colorjitter', 'gaussiannoise', 'gaussianblur', 'elastic']
        self.random_operations_unlabeled_weak = ["vertical", "horizontal", "rotation"] # CHANGE removed "scaling"
        self.random_operations_unlabeled_strong = ["colorjitter", "noise", "blur"] # Try without elastic deformation first
        self.gaussian_blur = v2.GaussianBlur(kernel_size=5, sigma=(0.0001,0.1))
        self.gaussian_noise = v2.GaussianNoise()
        self.elastic = Rand2DElasticd(
            keys=['image', 'target'],
            prob=1.0,
            spacing=(80, 80),              # σ ≈ 9–11 → use 10 px control-point spacing
            magnitude_range=(10, 20),     # α ∈ [80, 120] CHANGE /4 since we have 512x512 patches
            rotate_range=(0.0,),           # no rotation (handled separately)
            scale_range=(0.0, 0.0),        # no scaling here
            translate_range=(0.0, 0.0),    # no translation
            padding_mode="reflection",
            mode=['bilinear', 'nearest'],
            device=None,                   # auto-matches CPU/GPU
        )
        self.unlabeled_noise = v2.GaussianNoise(0.1)





        # KORNIA AUGMENTATIONS
        self.nr_weak_transforms = 2
        self.nr_strong_transforms = 2
        self.nr_labeled_transforms = 4
        self.kornia_weak = [K.RandomHorizontalFlip(p=1), K.RandomVerticalFlip(p=1), K.RandomRotation90(times=(-3,3), p=1)]
        # self.kornia_strong = [K.ColorJiggle(brightness=(0.65,1.35), contrast=(0.5,1.5), saturation=(0.9,1.1), hue=(-0.1,0.1), p=1), K.RandomElasticTransform(sigma=(20,20), alpha=(3,3), p=1,padding_mode="reflection"), K.RandomGaussianBlur(kernel_size=(7,7), sigma=(0.1,1), p=1), K.RandomGaussianNoise(mean=0, std=0.1, p=1)]
        self.kornia_strong = [K.ColorJiggle(brightness=(0.9,1.1), contrast=(0.9,1.1), saturation=(0.95,1.05), hue=(-0.05,0.05), p=1), K.RandomGaussianNoise(mean=0, std=0.1, p=1)]
        self.kornia_all = self.kornia_weak + self.kornia_strong

    def unlabeled_transform_kornia(self,image):
        image = self.imgtotensor(image) # Output: (C, H, W) in [0, 255]
        image = image.float() / 255.0
        image = image.cuda()

        n_weak_ops = random.randint(0,self.nr_weak_transforms)
        if n_weak_ops > 0:
            chosen_weak = random.sample(self.kornia_weak, n_weak_ops)
            weak_seq = K.AugmentationSequential(*chosen_weak, data_keys=["input"], keepdim=True)
            image = weak_seq(image)
        
        strong_aug_image = image.clone()
        n_strong_ops = random.randint(1,self.nr_strong_transforms)
        chosen_strong = random.sample(self.kornia_strong, n_strong_ops)
        strong_seq = K.AugmentationSequential(*chosen_strong, data_keys=["input"], keepdim=True)
        strong_aug_image = strong_seq(strong_aug_image)
        
        return image, strong_aug_image
    
    def labeled_transform_kornia(self,image,target,ds_type):
        image = self.imgtotensor(image) # Output: (C, H, W) in [0, 255]
        image = image.float() / 255.0
        target = torch.from_numpy(target).long()
        image,target = image.cuda(), target.cuda()

        if ds_type == "train":
            n_ops = random.randint(0,self.nr_labeled_transforms)
            if n_ops > 0:
                chosen_ops = random.sample(self.kornia_all, n_ops)
                aug_seq = K.AugmentationSequential(*chosen_ops, data_keys=["input", "mask"], keepdim=True)
                image, target = aug_seq(image, target)
        
        return image, target

    def zoom_in(image, target, min_zoom=1.0, max_zoom=1.3):
        """
        image: tensor (C,H,W) or PIL
        target: tensor (H,W) segmentation mask
        min_zoom: 1.0 means no zoom, >1.0 means zoom in (crop)
        """
        _, h, w = image.shape
        factor = random.uniform(min_zoom, max_zoom)

        # Compute crop size (smaller area -> zoom-in effect)
        new_h = int(h / factor)
        new_w = int(w / factor)

        # Random top-left corner (so crop is not always centered)
        if h == new_h or w == new_w:
            top, left = 0, 0
        else:
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)

        # Crop smaller region and resize back to original size
        image = F.resized_crop(
            image, top, left, new_h, new_w, (h, w),
            interpolation=F.InterpolationMode.BILINEAR # Should be faster
        )
        target = F.resized_crop(
            target, top, left, new_h, new_w, (h, w),
            interpolation=F.InterpolationMode.NEAREST
        )

        return image, target
    
    def custom_transform_v3(self, image, target, ds_type):
        image = self.imgtotensor(image) # Output: (C, H, W) in [0, 255]
        image = image.float() / 255.0
        target = torch.from_numpy(target).long()

        if ds_type == 'train':
            n_ops = random.randint(0,self.N_OPS)
            if n_ops == 0:
                return image,target
            else:
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
                    elif operation =='scaling':
                        image,target = self.zoom_in(image=image,target=target)
                    elif operation == 'colorjitter':
                        image = self.jitter(image)
                    elif operation == 'gaussiannoise':
                        image = self.gaussian_noise(image)
                    elif operation == 'gaussianblur':
                        image = self.gaussian_blur(image)
                    elif operation == 'elastic':
                        seed = random.randint(0, 99999)
                        self.elastic.set_random_state(seed)
                        data_dict = {'image': image, 'target': target.unsqueeze(0)}
                        resul_dict = self.elastic(data_dict) # Pass the dictionary
                        image,target = resul_dict['image'],resul_dict['target'].squeeze(0).long()
        return image,target




    def unlabeled_transform(self,image):
        image = self.imgtotensor(image) # Output: (C, H, W) in [0, 255]
        image = image.float() / 255.0

        weak_transformation = random.choice(self.random_operations_unlabeled_weak)
        if weak_transformation == "rotation":
            degree = random.choice([-90,-180,-270,90,180,270])
            image = F.rotate(image, degree)
        elif weak_transformation == "vertical":
            image = F.vflip(image)
        elif weak_transformation == "horizontal":
            image = F.hflip(image)
        noisy_image = image.clone()

        n_ops = random.randint(1,self.N_OPS_UNLABELED)
        for n in range(n_ops):
            strong_transformation = random.choice(self.random_operations_unlabeled_strong)
            if strong_transformation  == "colorjitter":
                noisy_image = self.jitter(noisy_image)
            elif strong_transformation == "blur":
                noisy_image = self.gaussian_blur(noisy_image)
            elif strong_transformation == "noise":
                noisy_image = self.gaussian_noise(noisy_image)

        return image, noisy_image

    def custom_transform(self, image, target, ds_type):
        # 1. Convert to Tensor
        image = self.imgtotensor(image) # Output: (C, H, W) in [0, 255]
        target = torch.from_numpy(target).long()
        
        # 2. Convert to TV Tensors (for correct transform pairing)
        # image = tv_tensors.Image(image)
        # target = tv_tensors.Mask(target)

        # 3. Apply Augmentations (Train Only)
        if ds_type == 'train':
            if random.random() > 0.5:
                image = F.hflip(image)
                target = F.hflip(target)
            if random.random() > 0.5:
                image = F.vflip(image)
                target = F.vflip(target)

            image = image.float() / 255.0
            
            # Color Jitter and Normalization on Jittered Branch
            if random.random() > 0.5:
                image = self.jitter(image)
                
        else: # ds_type == 'test' or 'val'
            # NORMALIZATION IS DONE HERE for test/val data
            image = image.float() / 255.0
        return image, target

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Happens once per Dataloader Worker

        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        
        key, wsi_id = self.metadata[idx]
        with self.env.begin() as txn:
            patch_data = txn.get(key.encode() + b"_patch")
            patch = Image.open(io.BytesIO(patch_data))
            if not self.unlabeled:
                target_data = txn.get(key.encode() + b"_target")
                target = imread(io.BytesIO(target_data))

        patch = self.imgtotensor(patch).float() / 255.0
        if not self.unlabeled:
            target = torch.from_numpy(target).long()  # CPU tensor
            if self.ds_type == 'train':
                if self.return_wsi_id:
                    return patch, target, wsi_id
                else:
                    return patch, target
            else:
                return patch, target, wsi_id
        else:
            if self.return_wsi_id:
                return patch, wsi_id
            else:
                return patch

        # if not self.unlabeled:
        #     # patch, target = self.custom_transform_v3(patch, target, self.ds_type)
        #     patch, target = self.labeled_transform_kornia(patch, target, self.ds_type)

        # # For unlabeled dataset just return the image
        # else:
        #     # patch, patch_noisy = self.unlabeled_transform(patch)
        #     patch, patch_noisy = self.unlabeled_transform_kornia(patch)
        #     if self.return_wsi_id:
        #         return patch, patch_noisy,wsi_id
        #     else:
        #         return patch,patch_noisy
        # if self.ds_type == 'train':
        #     if self.return_wsi_id:
        #         return patch,target,wsi_id
        #     else:
        #         return patch,target
        # else:
        #     return patch,target,wsi_id

    def get_labeled_category_counts(self):
        categories = np.array([cat for _, cat in self.labeled_categories])
        return np.bincount(categories, minlength=2) # THIS WAS IT
    

    def get_wsi_ids(self):
        # Don't have to convert strings to int. Gets handled by CV
        return [wsi_id for _, wsi_id in self.metadata]

    def __del__(self):
        if self.env:
            self.env.close()


def create_balanced_dataloader(
    dataset: Dataset | Subset,
    num_workers:int,
    total_images: int = None,
    batch_size=32,
    replacement=True,
    hpc = True
):
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        subset_indices = dataset.indices
        subset_labeled_metadata = [
            original_dataset.labeled_categories[i] for i in subset_indices
        ]

        labeled_categories = np.array([cat for _, cat in subset_labeled_metadata])
        category_counts = np.bincount(labeled_categories, minlength=2) # CHANGE  to minlength 2 from 3 previously

        # Calculate sample weights (assigning a weight to EACH SAMPLE in the subset)
        # This requires mapping the category back to the subset's samples
        weights = 1.0 / np.maximum(category_counts, 1)
        sample_weights = np.array([weights[cat] for cat in labeled_categories])

    else:
        # If it's the original LMDataset object, use the original methods
        category_counts = dataset.get_labeled_category_counts()
        weights = 1.0 / np.maximum(category_counts, 1)
        sample_weights = np.array(
            [weights[cat] for _, cat in dataset.labeled_categories]
        )

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