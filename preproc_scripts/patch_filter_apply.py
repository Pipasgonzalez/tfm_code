import numpy as np
from typing import Optional, List
from scipy.stats import entropy

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import cv2
from histomicstk.preprocessing.color_deconvolution import color_deconvolution
import json
import lmdb
import io
from imageio.v3 import imread
import joblib
from loguru import logger
from typing import Literal

from src.helpers import CLASS_MAPPING

CLASS_IDS = sorted(np.unique(list(CLASS_MAPPING.values())))
GLOBAL_CATEGORY_COUNTS = {0: 0, 1: 0, 2: 0}


# Threshholds etc. from training classifiers and testing them
CONFIG = json.load(open("./src/config/config.json"))


def batch_filter_unlabeled(
    features: np.ndarray, classifier, scaler=StandardScaler
) -> int:
    scaled_features = scaler.transform(features)
    results_proba = classifier.predict_proba(scaled_features)[:, 1]
    results = (results_proba >= CONFIG["unlabeled_threshhold"]).astype(np.uint8)
    return results


def batch_filter_labeled(
    targets: List[np.ndarray],
    class_ids_len: int = 4,
    desmoplastic_idx: int = 1,
    entropy_cutoff: float = 0.5,#CONFIG["entropy_cutoff"],
    desmo_threshold: float = 0.25,  # ← NEW threshold fraction (10%),
) -> List[Optional[int]]:
    """
    Vectorized processing of a batch of target arrays to assign categories (0, 1, or None).

    Args:
        targets: List of numpy arrays (e.g., label masks) to classify.
        class_ids_len: Length of CLASS_IDS for bincount.
        desmoplastic_idx: Index of 'desmoplastic_ring_effective' in CLASS_MAPPING.
        entropy_cutoff: Entropy threshold from CONFIG.

    Returns:
        List of category labels (0, 1, or 2) for each target.
    """

    # Stack valid targets and flatten each for bincount
    flattened_targets = [t.flatten() for t in targets]

    # Compute class counts for all valid targets at once
    # Create a 2D array where each row is the bincount for one target
    class_counts = np.array(
        [np.bincount(t, minlength=class_ids_len) for t in flattened_targets]
    )

    # CHANGE Added Percentage Based Filtering of dHGP Patches
    total_pixels = class_counts.sum(axis=1, keepdims=True)
    percentage_class_counts = class_counts / total_pixels

    # Check for desmoplastic_ring_effective (category 0)
    # desmoplastic_present = class_counts[:, desmoplastic_idx] > 0
    desmoplastic_present = percentage_class_counts[:, desmoplastic_idx] >= desmo_threshold

    # Compute entropy for all targets where desmoplastic is not present
    # entropies = np.array(
    #     [
    #         entropy(counts, base=2) if not desmo else 0.0
    #         for counts, desmo in zip(class_counts, desmoplastic_present)
    #     ]
    # )

    entropies = np.array(
        [
            entropy(counts, base=2)
            for counts, desmo in zip(class_counts, desmoplastic_present)
        ]
    )

    # Assign categories
    results_valid = [
        # 0 if desmo else (1 if ent > entropy_cutoff else 2)
        1 if ent > entropy_cutoff else 2
        for desmo, ent in zip(desmoplastic_present, entropies)
    ]
    global GLOBAL_CATEGORY_COUNTS
    
    # Iterate through the results and update the count for each category
    for result in results_valid:
        # Check if the result is one of the categories we are counting
        if result in GLOBAL_CATEGORY_COUNTS:
            GLOBAL_CATEGORY_COUNTS[result] += 1
    return results_valid


def main(
    res_level:int,
    labeled_filter: bool = False,
    unlabeled_filter: bool = False,
    lmdb_path: str = "D:/tfm_data/preprocessed/lmdb_raw",
    batch_size: int = 500,
):
    classifier = joblib.load("./src/pretrained_models/rf_lvl5.joblib")
    scaler = joblib.load("./src/pretrained_models/scaler_lvl5.joblib")

    MAPPING_DATASET_SIZES = {
        "3": 30,
        "4": 8,
        "5": 2, # CHANGE for unlabeled ds
        "6": 1
    }

    # 28215.11 MB for lvl 3
    # 7040.95 MB for lvl 4
    # 1711.79 MB for lvl 5
    gb = MAPPING_DATASET_SIZES.get(str(res_level))
    if gb is None:
        raise ValueError("Map size is none")
    map_size =  gb * 1024 * 1024 * 1024
    env = lmdb.open(lmdb_path, lock=False, map_size = map_size)
    with env.begin() as txn:
        metadata_buffer = txn.get(b"__metadata")
        if metadata_buffer is None:
            try:
                metadata_array = np.load("./src/config/metadata_lvl5_labeled.npz", allow_pickle=True)
            except Exception as e:
                logger.info(f"Could not even load numpy metadata {str(e)}")
        else:
            metadata_array = np.load(io.BytesIO(metadata_buffer), allow_pickle=True)
        metadata = list(zip(metadata_array["keys"], metadata_array["wsi_ids"]))

    batch = []
    batch_keys = []
    counter = 0

    labeled_result = []
    unlabeled_result = []
    with env.begin(write=True) as txn:
        for key, wsi_id in metadata:
            curr_patch = txn.get(key + b"_patch")
            curr_target = txn.get(key + b"_target")
            curr_features = txn.get(key + b"_features")

            # patch_np = imread(io.BytesIO(curr_patch))
            target_np = imread(io.BytesIO(curr_target))
            features_np = np.frombuffer(io.BytesIO(curr_features).getvalue())

            batch.append((target_np, features_np))
            batch_keys.append(key.decode())
            counter += 1
            if counter % batch_size == 0 or counter == len(metadata):
                logger.info(f"Pushing Batch at IDX {counter}")
                masks, features_batch = zip(*batch)
                features_batch = np.vstack(features_batch)
                if labeled_filter:
                    labeled_batch = batch_filter_labeled(
                        masks,
                        class_ids_len=len(CLASS_IDS),
                        desmoplastic_idx=CLASS_MAPPING["desmoplastic_ring_effective"],
                        #entropy_cutoff=CONFIG["entropy_cutoff"],
                    )
                    labeled_result.extend(list(zip(batch_keys, labeled_batch)))
                if unlabeled_filter:
                    unlabeled_batch = batch_filter_unlabeled(
                        features_batch, classifier, scaler
                    )
                    unlabeled_result.extend(list(zip(batch_keys, unlabeled_batch)))
                batch_keys.clear()
                batch.clear()

        # Clean Up Remaining Batches
        if len(batch) > 0:
            masks, features_batch = zip(*batch)
            features_batch = np.vstack(features_batch)
            if labeled_filter:
                labeled_batch = batch_filter_labeled(
                    masks,
                    class_ids_len=len(CLASS_IDS),
                    desmoplastic_idx=CLASS_MAPPING["desmoplastic_ring_effective"],
                    #entropy_cutoff=CONFIG["entropy_cutoff"],
                )
                labeled_result.extend(list(zip(batch_keys, labeled_batch)))
            if unlabeled_filter:
                unlabeled_batch = batch_filter_unlabeled(
                    features_batch, classifier, scaler
                )
                unlabeled_result.extend(list(zip(batch_keys, unlabeled_batch)))
            batch_keys.clear()
            batch.clear()
    with env.begin(write=True) as txn:
        if labeled_filter:
            labeled_buffer = io.BytesIO()
            np.savez_compressed(
                labeled_buffer,
                keys=np.array([k for k, _ in labeled_result], dtype="S8"),
                category=np.array([c for _, c in labeled_result], dtype=np.uint8),
            )
            txn.put(b"__labeled_categories", labeled_buffer.getvalue())
        if unlabeled_filter:
            unlabeled_buffer = io.BytesIO()
            np.savez_compressed(
                unlabeled_buffer,
                keys=np.array([k for k, _ in unlabeled_result], dtype="S8"),
                category=np.array([c for _, c in unlabeled_result], dtype=np.uint8),
            )
            txn.put(b"__unlabeled_categories", unlabeled_buffer.getvalue())
    global GLOBAL_CATEGORY_COUNTS
    print(GLOBAL_CATEGORY_COUNTS)

def main_unlabeled(
    res_level:int,
    labeled_filter: bool = False,
    unlabeled_filter: bool = False,
    lmdb_path: str = "D:/tfm_data/preprocessed/lmdb_raw",
    batch_size: int = 500,
):
    classifier = joblib.load("./src/pretrained_models/rf_unlabeled.joblib")
    scaler = joblib.load("./src/pretrained_models/rf_unlabeled_scaler.joblib")

    MAPPING_DATASET_SIZES = {
        "3": 30,
        "4": 8,
        "5": 4, # CHANGE for unlabeled ds
        "6": 1
    }

    # 28215.11 MB for lvl 3
    # 7040.95 MB for lvl 4
    # 1711.79 MB for lvl 5
    gb = MAPPING_DATASET_SIZES.get(str(res_level))
    if gb is None:
        raise ValueError("Map size is none")
    map_size =  gb * 1024 * 1024 * 1024
    env = lmdb.open(lmdb_path, lock=False, map_size = map_size)
    with env.begin() as txn:
        metadata_buffer = txn.get(b"__metadata")
        if metadata_buffer is None:
            raise ValueError("No metadata found in LMDB")
        metadata_array = np.load(io.BytesIO(metadata_buffer), allow_pickle=True)
        metadata = list(zip(metadata_array["keys"], metadata_array["wsi_ids"]))

    batch = []
    batch_keys = []
    counter = 0

    labeled_result = []
    unlabeled_result = []
    with env.begin(write=True) as txn:
        for key, wsi_id in metadata:
            # curr_patch = txn.get(key + b"_patch")
            curr_target = txn.get(key + b"_target")
            curr_features = txn.get(key + b"_features")

            # patch_np = imread(io.BytesIO(curr_patch))
            # target_np = imread(io.BytesIO(curr_target))
            features_np = np.frombuffer(io.BytesIO(curr_features).getvalue())

            # batch.append((target_np, features_np))
            batch.append((features_np))
            batch_keys.append(key.decode())
            counter += 1
            if counter % batch_size == 0 or counter == len(metadata):
                logger.info(f"Pushing Batch at IDX {counter}")
                # masks, features_batch = zip(*batch)
                # features_batch = zip(*batch)
                features_batch = np.vstack(batch)
                if labeled_filter:
                    labeled_batch = batch_filter_labeled(
                        masks,
                        class_ids_len=len(CLASS_IDS),
                        desmoplastic_idx=CLASS_MAPPING["desmoplastic_ring_effective"],
                        # entropy_cutoff=CONFIG["entropy_cutoff"],
                    )
                    labeled_result.extend(list(zip(batch_keys, labeled_batch)))
                if unlabeled_filter:
                    unlabeled_batch = batch_filter_unlabeled(
                        features_batch, classifier, scaler
                    )
                    unlabeled_result.extend(list(zip(batch_keys, unlabeled_batch)))
                batch_keys.clear()
                batch.clear()

        # Clean Up Remaining Batches
        if len(batch) > 0:
            masks, features_batch = zip(*batch)
            features_batch = np.vstack(features_batch)
            if labeled_filter:
                labeled_batch = batch_filter_labeled(
                    masks,
                    class_ids_len=len(CLASS_IDS),
                    desmoplastic_idx=CLASS_MAPPING["desmoplastic_ring_effective"],
                    #entropy_cutoff=CONFIG["entropy_cutoff"],
                )
                labeled_result.extend(list(zip(batch_keys, labeled_batch)))
            if unlabeled_filter:
                unlabeled_batch = batch_filter_unlabeled(
                    features_batch, classifier, scaler
                )
                unlabeled_result.extend(list(zip(batch_keys, unlabeled_batch)))
            batch_keys.clear()
            batch.clear()
    with env.begin(write=True) as txn:
        if labeled_filter:
            labeled_buffer = io.BytesIO()
            np.savez_compressed(
                labeled_buffer,
                keys=np.array([k for k, _ in labeled_result], dtype="S8"),
                category=np.array([c for _, c in labeled_result], dtype=np.uint8),
            )
            txn.put(b"__labeled_categories", labeled_buffer.getvalue())
        if unlabeled_filter:
            unlabeled_buffer = io.BytesIO()
            np.savez_compressed(
                unlabeled_buffer,
                keys=np.array([k for k, _ in unlabeled_result], dtype="S8"),
                category=np.array([c for _, c in unlabeled_result], dtype=np.uint8),
            )
            txn.put(b"__unlabeled_categories", unlabeled_buffer.getvalue())
    global GLOBAL_CATEGORY_COUNTS
    print("Done performing preprocessing. Labeled Category Distributions:", GLOBAL_CATEGORY_COUNTS)

# Helper
import shutil
import os

def copy_lmdb(src_path: str, dst_path: str, overwrite: bool = False):
    """
    Copy an LMDB dataset folder (with all contents) to a new location.
    Works whether src_path contains a single LMDB or multiple.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source LMDB folder not found: {src_path}")

    if os.path.exists(dst_path):
        if overwrite:
            print(f"⚠️  Destination exists, overwriting: {dst_path}")
            shutil.rmtree(dst_path)
        else:
            raise FileExistsError(f"Destination LMDB already exists: {dst_path}")

    # Copy the entire folder tree
    shutil.copytree(src_path, dst_path)

if __name__ == "__main__":
    # Copy the original lmdb
    # print('Copying DS')
    # copy_lmdb(src_path="D:/tfm_data/preprocessed/lmdb_raw",dst_path="D:/tfm_data/preprocessed/small_dHGP_corrected", overwrite=True)
    main(labeled_filter=True, unlabeled_filter=False,lmdb_path="D:/tfm_data/preprocessed/lvl5_labeled_normalized", res_level=5)

# For Level 5 labeled {0: 195, 1: 2469, 2: 1073}