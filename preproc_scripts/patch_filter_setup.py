# python -m src.preproc_scripts.unlabeled_patch_filtering.py
import lmdb
import zlib
from imageio.v3 import imread
import io
import numpy as np
from histomicstk.preprocessing.color_deconvolution import color_deconvolution
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from scipy.spatial import cKDTree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import joblib
from loguru import logger
from pathlib import Path
import os

from src.utils.data import LMDBTorchDataset

def create_patch_features(
    ds_path: str = "D:/tfm_data/preprocessed/lmdb_raw",
    w=[[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]],
    output_filename="./src/data/patch_features/features.npz",
    data_limit=None,
):
    if not Path(output_filename).parent.is_dir():
        raise ModuleNotFoundError()

    # Helper Functions
    def subpatch_hema_densities(hema, sub_size=32):
        h, w = hema.shape
        _, nuclei_mask = cv2.threshold(
            hema, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        patches = [
            nuclei_mask[i : i + sub_size, j : j + sub_size]
            for i in range(0, h, sub_size)
            for j in range(0, w, sub_size)
        ]
        return [np.mean(nuclei_mask > 0) for nuclei_mask in patches], nuclei_mask

    def subpatch_eosin_densities(eosi: np.ndarray, sub_size=32):
        h, w = eosi.shape
        patches = [
            eosi[i : i + sub_size, j : j + sub_size]
            for i in range(0, h, sub_size)
            for j in range(0, w, sub_size)
        ]
        return [np.mean(patch) for patch in patches]
    
    def get_macro_features(densities):
        # Reshape your flat list of 256 densities back into a 16x16 grid
        # (Since 512 / 32 = 16)
        #Instead of just calculating the mean density of the whole patch, split the patch into a grid (e.g., 2x2 or 4x4). Calculate the mean density for each grid cell, and then take the standard deviation of those means.
        # Logic: If a patch is pure liver, the density in the top-left corner will be roughly the same as the bottom-right. The std dev of the grid will be low. If the patch is half-liver/half-tumor, one quadrant will be dense, the other sparse. The std dev will be high.
        grid = np.array(densities).reshape(16, 16)
        
        # Calculate Left vs Right and Top vs Bottom differences
        left_mean = np.mean(grid[:, :8])
        right_mean = np.mean(grid[:, 8:])
        top_mean = np.mean(grid[:8, :])
        bottom_mean = np.mean(grid[8:, :])
        
        # Structural Heterogeneity features
        horizontal_split = np.abs(left_mean - right_mean)
        vertical_split = np.abs(top_mean - bottom_mean)
        return np.array([horizontal_split, vertical_split])
    

    def get_hema_features(densities: np.ndarray, hema_channel: np.ndarray):
        mean_density = np.mean(densities, axis=0)
        variance_density = np.var(densities, axis=0)
        skewness_density = skew(densities, axis=0)
        kurtosis_density = kurtosis(densities, axis=0)

        percentile_25 = np.percentile(densities, 25)
        percentile_50 = np.percentile(densities, 50)
        percentile_75 = np.percentile(densities, 75)

        glcm = graycomatrix(
            hema_channel,
            distances=[5],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        contrast = np.mean(graycoprops(glcm, "contrast"))
        homogeneity = np.mean(graycoprops(glcm, "homogeneity"))
        energy = np.mean(graycoprops(glcm, "energy"))
        correlation = np.mean(graycoprops(glcm, "correlation"))

        feature_vector = np.array(
            [
                mean_density,
                variance_density,
                skewness_density,
                kurtosis_density,
                percentile_25,
                percentile_50,
                percentile_75,
                contrast,
                homogeneity,
                energy,
                correlation,
            ]
        )
        return feature_vector

    def get_eosin_features(eosin_densities: np.ndarray):
        mean_density = np.mean(eosin_densities, axis=0)
        variance_density = np.var(eosin_densities, axis=0)
        skewness_density = skew(eosin_densities, axis=0)
        kurtosis_density = kurtosis(eosin_densities, axis=0)
        percentile_25 = np.percentile(eosin_densities, 25)
        percentile_50 = np.percentile(eosin_densities, 50)
        percentile_75 = np.percentile(eosin_densities, 75)
        return np.array(
            [
                mean_density,
                variance_density,
                skewness_density,
                kurtosis_density,
                percentile_25,
                percentile_50,
                percentile_75,
            ]
        )

    def get_morphological_features(nuclei_mask: np.ndarray):
        # Find all distinct objects and calculate their properties
        # The output 'stats' is a matrix where each row is an object
        # and columns are [left, top, width, height, area]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            nuclei_mask, connectivity=4 # CHANGE dropped from 8 for computational gain
        )

        # Ignore the first label (0), which is the background
        if num_labels <= 1:
            # Return a zero vector if no nuclei are found
            return np.zeros(6)

        areas = stats[1:, cv2.CC_STAT_AREA]
        valid_centroids = centroids[1:]
        object_count = num_labels - 1

        # 2. Statistics of the areas
        mean_area = np.mean(areas)
        var_area = np.var(areas)

        feature_vector = np.array(
            [
                object_count,
                mean_area,
                var_area,
                np.max(areas),
                np.min(areas),
                np.median(areas),
            ]
        )

        return feature_vector, valid_centroids
    def get_spatial_features(centroids):
        """
        Calculates statistics on Nearest Neighbor distances.
        Input: Array of (x,y) coordinates.
        """
        # We need at least 3 points to calculate variance/skew meaningfully
        if centroids is None or len(centroids) < 3:
            return np.zeros(4) # Return 0s if not enough nuclei

        # Build the Tree (Extremely fast for <1000 points)
        tree = cKDTree(centroids)
        
        # Query the distance to the nearest neighbor (k=2)
        # k=1 is the point itself (dist=0), so we take the second one.
        # workers=-1 uses all CPU cores, but for small N, workers=1 is actually faster due to overhead.
        dists, _ = tree.query(centroids, k=2, workers=1) 
        
        # The result is [dist_to_self, dist_to_neighbor], we want column 1
        nn_dists = dists[:, 1]

        return np.array([
            np.mean(nn_dists),  # Avg spacing (dense vs sparse)
            np.var(nn_dists),   # Variance (uniform vs clustered) -> KEY for heterogeneity
            skew(nn_dists),     # Skewness of spacing
            kurtosis(nn_dists)  # Kurtosis of spacing
        ])
    # Process Dataset
    c = 0
    all_features = []
    all_labels = []
    all_keys = []
    features_dict = {}

    DATA_LIMIT = data_limit or 9999999999999
    with lmdb.open(ds_path, readonly=True) as env:
        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    if c >= DATA_LIMIT:
                        break
                    if b"_patch" in key:
                        c += 1
                        curr_patch = value
                        str_key = key.decode().split("_")[0]
                        target_key = (str_key + "_target").encode()
                        curr_target = txn.get(target_key)
                        patch_np = imread(io.BytesIO(curr_patch))
                        target_np = imread(io.BytesIO(curr_target))

                        if c % 1000 == 0:
                            logger.info(f"Reached Patch IDX {c}")

                        # --- Label Assignment ---
                        bin_counts = np.bincount(target_np.flatten(), minlength=4)
                        total_pixels = np.sum(bin_counts)
                        if total_pixels == 0:
                            continue  # Skip empty masks

                        max_percentage = np.max(bin_counts) / total_pixels

                        # --- Feature Calculation ---
                        h_e = color_deconvolution(patch_np, w)
                        hema = h_e.Stains[:, :, 0].astype(np.uint8)
                        eosin = h_e.Stains[:, :, 1].astype(np.uint8)

                        # Check if hematoxylin channel is valid
                        if np.max(hema) == 0:
                            continue

                        hematoxylin_densities, nuclei_mask = subpatch_hema_densities(hema)
                        eosin_densities = subpatch_eosin_densities(eosin)
                        correlation = np.corrcoef(
                            np.array(hematoxylin_densities).flatten(),
                            np.array(eosin_densities).flatten(),
                        )[0, 1]

                        hematoxylin_features = get_hema_features(hematoxylin_densities, hema)
                        eosin_features = get_eosin_features(eosin_densities)
                        morph_features, valid_centroids = get_morphological_features(nuclei_mask)
                
                        # Macro Features
                        hema_macro = get_macro_features(hematoxylin_densities)
                        eosin_macro = get_macro_features(eosin_densities)
                        # Spatial Features
                        spatial_features = get_spatial_features(valid_centroids)
                        # --- Append Features and Labels ---

                        features = np.concatenate(
                            [
                                hematoxylin_features,
                                eosin_features,
                                [correlation],
                                morph_features,
                                hema_macro,
                                eosin_macro,
                                spatial_features
                            ]
                        )
                        all_features.append(features)
                        all_keys.append(str_key) # CHANGE added
                        features_dict[str_key] = features
                        if max_percentage >= 0.99:
                            # Assign class 0 for "Pure"
                            all_labels.append(0)
                        else:
                            # Assign class 1 for "Multi-class"
                            all_labels.append(1)
    X = np.array(all_features)
    y = np.array(all_labels)
    np.savez(output_filename, features=X, labels=y, keys = np.array(all_keys))
    # with lmdb.open(ds_path) as env:
    #     with env.begin(write=True) as txn:
    #         for key, features in features_dict.items():
    #             new_key = (key + "_features").encode()
    #             txn.put(new_key, features.tobytes())


def create_patch_features_unlabeled(
    ds_path: str = "D:/tfm_data/preprocessed/lmdb_raw",
    w=[[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]],
    output_filename="./src/data/patch_features/features.npz",
    data_limit=None,
):
    if not Path(output_filename).parent.is_dir():
        raise

    # Helper Functions
    def subpatch_densities(hema, sub_size=32):
        h, w = hema.shape
        _, nuclei_mask = cv2.threshold(
            hema, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        patches = [
            nuclei_mask[i : i + sub_size, j : j + sub_size]
            for i in range(0, h, sub_size)
            for j in range(0, w, sub_size)
        ]
        return [np.mean(nuclei_mask > 0) for nuclei_mask in patches], nuclei_mask

    def subpatch_eosin_densities(eosi: np.ndarray, sub_size=32):
        h, w = eosi.shape
        patches = [
            eosi[i : i + sub_size, j : j + sub_size]
            for i in range(0, h, sub_size)
            for j in range(0, w, sub_size)
        ]
        return [np.mean(patch) for patch in patches]

    def get_features(densities: np.ndarray, hema_channel: np.ndarray):
        mean_density = np.mean(densities, axis=0)
        variance_density = np.var(densities, axis=0)
        skewness_density = skew(densities, axis=0)
        kurtosis_density = kurtosis(densities, axis=0)

        percentile_25 = np.percentile(densities, 25)
        percentile_50 = np.percentile(densities, 50)
        percentile_75 = np.percentile(densities, 75)

        glcm = graycomatrix(
            hema_channel,
            distances=[5],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        contrast = np.mean(graycoprops(glcm, "contrast"))
        homogeneity = np.mean(graycoprops(glcm, "homogeneity"))
        energy = np.mean(graycoprops(glcm, "energy"))
        correlation = np.mean(graycoprops(glcm, "correlation"))

        feature_vector = np.array(
            [
                mean_density,
                variance_density,
                skewness_density,
                kurtosis_density,
                percentile_25,
                percentile_50,
                percentile_75,
                contrast,
                homogeneity,
                energy,
                correlation,
            ]
        )
        return feature_vector

    def get_eosin_features(eosin_densities: np.ndarray):
        mean_density = np.mean(eosin_densities, axis=0)
        variance_density = np.var(eosin_densities, axis=0)
        skewness_density = skew(eosin_densities, axis=0)
        kurtosis_density = kurtosis(eosin_densities, axis=0)
        percentile_25 = np.percentile(eosin_densities, 25)
        percentile_50 = np.percentile(eosin_densities, 50)
        percentile_75 = np.percentile(eosin_densities, 75)
        return np.array(
            [
                mean_density,
                variance_density,
                skewness_density,
                kurtosis_density,
                percentile_25,
                percentile_50,
                percentile_75,
            ]
        )

    def get_morphological_features(nuclei_mask: np.ndarray):
        # Find all distinct objects and calculate their properties
        # The output 'stats' is a matrix where each row is an object
        # and columns are [left, top, width, height, area]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            nuclei_mask, connectivity=8
        )

        # Ignore the first label (0), which is the background
        if num_labels <= 1:
            # Return a zero vector if no nuclei are found
            return np.zeros(6)

        areas = stats[1:, cv2.CC_STAT_AREA]

        object_count = num_labels - 1

        # 2. Statistics of the areas
        mean_area = np.mean(areas)
        var_area = np.var(areas)

        feature_vector = np.array(
            [
                object_count,
                mean_area,
                var_area,
                np.max(areas),
                np.min(areas),
                np.median(areas),
            ]
        )

        return feature_vector

    # Process Dataset
    c = 0
    all_features = []
    all_labels = []
    features_dict = {}

    DATA_LIMIT = data_limit or 9999999999999
    with lmdb.open(ds_path, readonly=True) as env:
        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    if c >= DATA_LIMIT:
                        break
                    if b"_patch" in key:
                        c += 1
                        curr_patch = value
                        str_key = key.decode().split("_")[0]
                        patch_np = imread(io.BytesIO(curr_patch))

                        if c % 1000 == 0:
                            logger.info(f"Reached Patch IDX {c}")

                        # --- Feature Calculation ---
                        h_e = color_deconvolution(patch_np, w)
                        hema = h_e.Stains[:, :, 0].astype(np.uint8)
                        eosin = h_e.Stains[:, :, 1].astype(np.uint8)

                        # Check if hematoxylin channel is valid
                        if np.max(hema) == 0:
                            continue

                        hematoxylin_densities, nuclei_mask = subpatch_densities(hema)
                        eosin_densities = subpatch_eosin_densities(eosin)
                        correlation = np.corrcoef(
                            np.array(hematoxylin_densities).flatten(),
                            np.array(eosin_densities).flatten(),
                        )[0, 1]

                        hematoxylin_features = get_features(hematoxylin_densities, hema)
                        eosin_features = get_eosin_features(eosin_densities)
                        morph_features = get_morphological_features(nuclei_mask)
                        # --- Append Features and Labels ---

                        features = np.concatenate(
                            [
                                hematoxylin_features,
                                eosin_features,
                                [correlation],
                                morph_features,
                            ]
                        )
                        all_features.append(features)
                        features_dict[str_key] = features

    X = np.array(all_features)
    np.savez(output_filename, features=X)
    # with lmdb.open(ds_path, map_size = 4 * 1024 * 1024 * 1024) as env:
    #     with env.begin(write=True) as txn:
    #         for key, features in features_dict.items():
    #             new_key = (key + "_features").encode()
    #             txn.put(new_key, features.tobytes())

def train_random_forest(
    data_filename: str = "./src/data/unlabeled_patch_features/features.npz",
):
    logger.info("Starting with RandomForest Fitting")
    rf = RandomForestClassifier(random_state=42, verbose=0)
    param_grid_rf = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [10, 20, 30, None],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", 0.5, 0.7],
    }
    random_search_rf = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid_rf,
        n_iter=30,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring="accuracy",
    )
    data = np.load(data_filename)
    X = data["features"]
    y = data["labels"]
    print(len(X))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    random_search_rf.fit(X_scaled, y)
    best_rf = random_search_rf.best_estimator_
    # Define the filename for the saved model
    model_filename = "./src/pretrained_models/rf_lvl5.joblib"
    scaler_filename = "./src/pretrained_models/scaler_lvl5.joblib"
    # Save the model
    joblib.dump(best_rf, model_filename)
    joblib.dump(scaler, scaler_filename)


def train_xgboost(
    data_filename: str = "./src/data/unlabeled_patch_features/features.npz",
):
    # XGBoost with Hyperparameter Tuning for Binary Classification (GPU-accelerated)
    data = np.load(data_filename)
    X = data["features"]
    y = data["labels"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    xgb = XGBClassifier(
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=10,
    )
    param_grid_xgb = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.5],
    }
    random_search_xgb = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid_xgb,
        n_iter=100,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=10,  # Set to 1 for GPU to avoid overhead
        scoring="roc_auc",
    )

    # Fit with early stopping
    random_search_xgb.fit(X_scaled, y, verbose=False)
    best_xgb = random_search_xgb.best_estimator_

def write_features_to_lmdb_from_npz(
    lmdb_path: str = "D:/tfm_data/preprocessed/dataset_lvl4",
    npz_path: str = "./src/data/patch_features/features_lvl4.npz",
):
    """
    Writes features from a saved NumPy .npz file back into an LMDB database.
    Optionally provide a .npy or .txt file with the keys (same order as features).
    """
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    ds = LMDBTorchDataset(lmdb_path=lmdb_path,ds_type='train', skip_val=True)
    keys = [met[0] for met in ds.metadata]
    assert len(keys) == len(features), f"Keys and features length mismatch: {len(keys)} vs {len(features)}"

    logger.info(f"Writing {len(keys)} feature vectors back into LMDB: {lmdb_path}")

    env = lmdb.open(lmdb_path, map_size=20*1024*1024*1024)
    with env.begin(write=True) as txn:
        for i, (key, feats) in enumerate(zip(keys, features)):
            lmdb_key = (key + "_features").encode()
            txn.put(lmdb_key, feats.tobytes())
            if i % 1000 == 0:
                logger.info(f"Wrote {i}/{len(keys)} feature vectors...")
    env.sync()
    env.close()
    logger.success("✅ Successfully wrote all features to LMDB.")

if __name__ == "__main__":
    # create_patch_features(ds_path="D:/tfm_data/preprocessed/dataset_lvl5_fmt", output_filename="./src/data/patch_features/features_lvl5.npz")
    # write_features_to_lmdb_from_npz()
    # logger.info("Training Random Forest")
    # train_random_forest(data_filename="./src/data/patch_features/features_lvl5.npz")
    create_patch_features(ds_path="D:/tfm_data/preprocessed/lvl4_black_removed_normalized", output_filename="c:/users/nicol/master_thesis/code/src/data/patch_features_lvl4_normalized_2711.npz")
    # create_patch_features_unlabeled(ds_path="D:/tfm_data/preprocessed/unlab_normalized_lvl4", output_filename="c:/users/nicol/master_thesis/code/src/data/patch_features_lvl4_normalized_unlabeled.npz")
