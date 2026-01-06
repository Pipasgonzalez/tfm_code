import lmdb
import numpy as np
import io
import zlib
import os
from typing import Optional
from src.helpers import CLASS_MAPPING
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import json
from src.utils.features import get_all_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# Logging
from loguru import logger

config = json.load(open("./src/config/config.json"))

# CLASS_MAPPING = {"background": 0, "dHGP": 1, "Liver": 2, "Tumor": 3}  # Example
# What would happen with these values --> run
# Theory: Last two classes: changing the percentage will maybe not change the dice score

import lmdb
import numpy as np
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
import joblib
GB_MAPPING = {
    "3": 31,
    "4": 8,
    "5": 2,
    "6": 1
}

GB_MAPPING_UNLABELED = {
    "3":25,
    "4":7,
    "5" : 1,
    "6": 0.5
}

class DatasetLMDB:
    def __init__(
        self, lmdb_path: str, num_wsi: int, resolution:int, batch_size: int = 250, max_workers: int = 1, 
    ):
        gb = GB_MAPPING.get(str(resolution),50)
        self.env = lmdb.open(
            lmdb_path,
            map_size=gb * 1024 * 1024 * 1024 , # 5s0 gb for now
            lock=False,
        )
        self.path = lmdb_path
        self.batch_size = batch_size
        self.batch = []
        self.metadata = []  # In-memory metadata list
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.resolution = resolution
        with self.env.begin() as txn:
            self.curr_idx = int(txn.get(b"__last_idx", b"0")) + 1

    def _compress_patch(self, patch: np.ndarray, target: np.ndarray):
        # Convert to uint 8
        patch = patch.astype(np.uint8)
        target = target.astype(np.uint8)
        patch_img = Image.fromarray(patch)
        target_img = Image.fromarray(target, mode="L")
        patch_buffer = io.BytesIO()
        target_buffer = io.BytesIO()
        patch_img.save(patch_buffer, format="PNG", compress_level=4)
        target_img.save(target_buffer, format="PNG", compress_level=4)
        return patch_buffer.getvalue(), target_buffer.getvalue()

    def save_patch(
        self,
        patch: np.ndarray,
        target: np.ndarray,
        wsi_id: str,
        coords
    ) -> bool:
        if (
            not isinstance(patch, np.ndarray)
            or not isinstance(target, np.ndarray)
            # or category is None
            or wsi_id is None
        ):
            return False
        self.batch.append((patch, target, wsi_id, coords))  # CHANGE added coords
        if len(self.batch) >= self.batch_size:
            self._flush_batch()
        return True

    def _flush_batch(self):
        if not self.batch:
            return
        compressed = list(
            self.executor.map(lambda x: self._compress_patch(x[0], x[1]), self.batch)
        )
        with self.env.begin(write=True) as txn:
            for (patch, target, wsi_id, coords), (patch_data, target_data) in zip(
                self.batch, compressed
            ):
                key = f"{self.curr_idx:08d}".encode()
                txn.put(key + b"_wsi_id", wsi_id.encode())
                txn.put(key + b"_patch", patch_data)
                txn.put(key + b"_target", target_data)

                coord_str = f"{coords[0]}_{coords[1]}"
                txn.put(key + b"_coords", coord_str.encode())
                # txn.put(key + b"_category", str(category).encode())
                txn.put(b"__last_idx", str(self.curr_idx).encode())
                self.metadata.append((key, wsi_id, coords[0], coords[1]))  # CHANGE added coordinates
                self.curr_idx += 1
            self.batch.clear()

    def save_metadata(self):
        with self.env.begin(write=True) as txn:
            metadata_buffer = io.BytesIO()
            keys = [row[0] for row in self.metadata]
            ids = [row[1] for row in self.metadata]
            xs = [row[2] for row in self.metadata] # New
            ys = [row[3] for row in self.metadata] # New
            np.savez_compressed(
                metadata_buffer,
                keys=np.array(keys, dtype="S8"),
                wsi_ids=np.array(ids, dtype="object"),
                coords_x=np.array(xs, dtype=np.int32), # Save X
                coords_y=np.array(ys, dtype=np.int32), # Save Y
            )
            txn.put(b"__metadata", metadata_buffer.getvalue())
            # Also save locally
            np.savez(
                file=f"./src/metadata_lvl{self.resolution}_labeled.npz",
                keys=np.array(keys, dtype="S8"),
                wsi_ids=np.array(ids, dtype="object"),
                coords_x=np.array(xs, dtype=np.int32), # Save X
                coords_y=np.array(ys, dtype=np.int32), # Save Y

            )

    def close(self):
        self._flush_batch()
        self.save_metadata()
        self.executor.shutdown()
        self.env.close()
        logger.info(
            f"Succesfully Completed Ingestion into Database LMDB File:{self.path}"
        )

    def get_metadata(self):
        """Retrieve all metadata from LMDB."""
        metadata = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b"__metadata_"):
                    buffer = io.BytesIO(value)
                    metadata_array = np.load(buffer)
                    metadata.extend(
                        [
                            (k, c)
                            for k, c in zip(
                                metadata_array["key"], metadata_array["category"]
                            )
                        ]
                    )
        return metadata


class UnlabeledDatasetLMDB:
    def __init__(
        self, lmdb_path: str, num_wsi: int, resolution:int, batch_size: int = 250, max_workers: int = 1, 
    ):
        gb = GB_MAPPING_UNLABELED.get(str(resolution),50)
        self.env = lmdb.open(
            lmdb_path,
            map_size=int((num_wsi/200)+1) * 8 * 1024 * 1024 * 1024, # 8GB per 200 WSI IDs (Experience Value)
            lock=False,
        )
        self.path = lmdb_path
        self.batch_size = batch_size
        self.batch = []
        self.metadata = []  # In-memory metadata list
        self.resolution = resolution
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        
        self.scaler:StandardScaler = joblib.load("./src/pretrained_models/scaler_lvl4.joblib")
        self.rf:RandomForestClassifier = joblib.load("./src/pretrained_models/rf_lvl4.joblib")
        with self.env.begin() as txn:
            self.curr_idx = int(txn.get(b"__last_idx", b"0")) + 1

        # self.features = []

    def _compress_patch(self, patch: np.ndarray):
        # Convert to uint 8
        patch = patch.astype(np.uint8)
        patch_img = Image.fromarray(patch)
        patch_buffer = io.BytesIO()
        patch_img.save(patch_buffer, format="PNG", compress_level=4)
        return patch_buffer.getvalue()

    def save_patch(
        self,
        patch: np.ndarray,
        wsi_id: str,
        coords
    ) -> bool:
        self.batch.append((patch, wsi_id, coords))
        if len(self.batch) >= self.batch_size:
            self._flush_batch()
        return True

    def _flush_batch(self):
        if not self.batch:
            return
    
        compressed = list(
            self.executor.map(lambda x: self._compress_patch(x[0]), self.batch) # Replaace self.batch with selected the last arg
        )
        with self.env.begin(write=True) as txn:
            for (patch, wsi_id, coords), (patch_data) in zip(
                self.batch, compressed # replace self.batch with selected
            ):
                key = f"{self.curr_idx:08d}".encode()
                txn.put(key + b"_wsi_id", wsi_id.encode())
                txn.put(key + b"_patch", patch_data)
                txn.put(b"__last_idx", str(self.curr_idx).encode())
                self.metadata.append((key, wsi_id, coords[0], coords[1]))
                self.curr_idx += 1

        self.batch.clear()

    def save_metadata(self):
        with self.env.begin(write=True) as txn:
            metadata_buffer = io.BytesIO()
            keys = [row[0] for row in self.metadata]
            ids = [row[1] for row in self.metadata]
            xs = [row[2] for row in self.metadata] # New
            ys = [row[3] for row in self.metadata] # New
            np.savez_compressed(
                metadata_buffer,
                keys=np.array(keys, dtype="S8"),
                wsi_ids=np.array(ids, dtype="object"),
                coords_x=np.array(xs, dtype=np.int32), # Save X
                coords_y=np.array(ys, dtype=np.int32), # Save Y
            )
            txn.put(b"__metadata", metadata_buffer.getvalue())
            np.savez(
                file=f"./src/config/metadata_lvl{self.resolution}_unlabeled.npz",
                keys=np.array(keys, dtype="S8"),
                wsi_ids=np.array(ids, dtype="object"),
                coords_x=np.array(xs, dtype=np.int32), # Save X
                coords_y=np.array(ys, dtype=np.int32), # Save Y
            )

    def close(self):
        self._flush_batch()
        # X = np.array(self.features)
        # np.savez("c:/users/nicol/master_thesis/code/src/data/patch_features_lvl4_normalized_unlabeled.npz", features=X)
        self.save_metadata()
        self.executor.shutdown()
        self.env.close()
        logger.info(
            f"Succesfully Completed Ingestion into Database LMDB File:{self.path}"
        )