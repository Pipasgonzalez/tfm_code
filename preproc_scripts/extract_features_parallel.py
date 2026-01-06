import lmdb
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from scipy.spatial import cKDTree
from histomicstk.preprocessing.color_deconvolution import color_deconvolution
from imageio.v3 import imread
import io
import multiprocessing as mp
from functools import partial
from loguru import logger
from pathlib import Path
import os

# --- 1. Optimized Feature Extraction Functions ---

def get_vectorized_subpatch_stats(image, sub_size=32):
    """
    Replaces the slow loop with numpy reshaping.
    Splits 512x512 -> 256 blocks of 32x32 in one go.
    """
    h, w = image.shape
    # Reshape into (rows, block_height, cols, block_width)
    # Then transpose to (rows, cols, block_height, block_width)
    # Then reshape to (num_blocks, block_pixels)
    n_rows = h // sub_size
    n_cols = w // sub_size
    
    # Create the blocks
    blocks = (image.reshape(n_rows, sub_size, n_cols, sub_size)
              .transpose(0, 2, 1, 3)
              .reshape(-1, sub_size * sub_size))
    
    # Calculate means of all blocks in one C-optimized call
    # For binary mask (nuclei), we want ratio of pixels > 0
    if image.dtype == bool or np.max(image) == 1:
        means = np.mean(blocks, axis=1)
    else:
        # For eosin density (grayscale intensity)
        means = np.mean(blocks, axis=1)
        
    return means

def get_nuclei_mask(hema):
    # Fast Otsu thresholding
    _, nuclei_mask = cv2.threshold(
        hema, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return nuclei_mask

def get_glcm_features_fast(image):
    """
    Optimized GLCM:
    1. Quantize image from 256 levels -> 32 levels (Speedup factor ~8x)
    2. Compute GLCM
    """
    # Binning: 0-255 -> 0-31
    image_binned = (image // 8).astype(np.uint8)
    
    glcm = graycomatrix(
        image_binned,
        distances=[5],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, # HUGE SPEEDUP vs 256
        symmetric=True,
        normed=True,
    )
    
    contrast = np.mean(graycoprops(glcm, "contrast"))
    homogeneity = np.mean(graycoprops(glcm, "homogeneity"))
    energy = np.mean(graycoprops(glcm, "energy"))
    correlation = np.mean(graycoprops(glcm, "correlation"))
    
    return np.array([contrast, homogeneity, energy, correlation])

def get_stats_moments(data):
    # Helper to clean up the main function
    return np.array([
        np.mean(data),
        np.var(data),
        skew(data),
        kurtosis(data),
        np.percentile(data, 25),
        np.percentile(data, 50),
        np.percentile(data, 75)
    ])

def get_macro_features(densities):
    # Reshape flat densities (256) back to 16x16 grid
    grid = densities.reshape(16, 16)
    left = np.mean(grid[:, :8])
    right = np.mean(grid[:, 8:])
    top = np.mean(grid[:8, :])
    bottom = np.mean(grid[8:, :])
    return np.array([np.abs(left - right), np.abs(top - bottom)])

def get_morph_and_spatial(nuclei_mask):
    # Combined to save repeated processing
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        nuclei_mask, connectivity=4
    )
    
    if num_labels <= 1:
        return np.zeros(6), np.zeros(4) # Morph(6), Spatial(4)

    # Morphological
    areas = stats[1:, cv2.CC_STAT_AREA]
    morph_feats = np.array([
        num_labels - 1,
        np.mean(areas),
        np.var(areas),
        np.max(areas),
        np.min(areas),
        np.median(areas)
    ])

    # Spatial (KDTree)
    centroids = centroids[1:]
    if len(centroids) < 3:
        spatial_feats = np.zeros(4)
    else:
        tree = cKDTree(centroids)
        # workers=1 is faster for small N
        dists, _ = tree.query(centroids, k=2, workers=1) 
        nn_dists = dists[:, 1]
        spatial_feats = np.array([
            np.mean(nn_dists),
            np.var(nn_dists),
            skew(nn_dists),
            kurtosis(nn_dists)
        ])
        
    return morph_feats, spatial_feats


# --- 2. The Worker Function (Runs on separate CPU core) ---

def process_chunk(keys_chunk, lmdb_path, w_matrix):
    """
    Opens its own LMDB connection and processes a list of keys.
    Returns a list of results: (key, feature_vector)
    """
    results = []
    
    # Re-open LMDB inside the worker process
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    try:
        with env.begin() as txn:
            for key_str in keys_chunk:
                # 1. Read Image
                lmdb_key = key_str.encode() + b"_patch"
                img_buffer = txn.get(lmdb_key)
                if img_buffer is None: continue
                
                patch_np = imread(io.BytesIO(img_buffer))
                
                # 2. Color Deconvolution
                # Check for white background (approx) to skip empty processing
                if np.mean(patch_np) > 240: 
                    # Optional: Skip entirely or return zeros? 
                    # Let's assume we process everything for safety
                    pass

                h_e = color_deconvolution(patch_np, w_matrix)
                hema = h_e.Stains[:, :, 0].astype(np.uint8)
                eosin = h_e.Stains[:, :, 1].astype(np.uint8)

                if np.max(hema) == 0: continue

                # 3. Calculate Features (Optimized)
                
                # A. Subpatch Densities (Vectorized)
                nuclei_mask = get_nuclei_mask(hema)
                hema_densities = get_vectorized_subpatch_stats(nuclei_mask) # fast
                eosin_densities = get_vectorized_subpatch_stats(eosin)      # fast
                
                # B. Correlation
                corr = np.corrcoef(hema_densities, eosin_densities)[0, 1]
                
                # C. Channel Stats
                hema_stats = get_stats_moments(hema_densities)
                hema_glcm = get_glcm_features_fast(hema) # Now fast
                hema_combined = np.concatenate([hema_stats, hema_glcm])
                
                eosin_combined = get_stats_moments(eosin_densities)
                
                # D. Macro Features
                hema_macro = get_macro_features(hema_densities)
                eosin_macro = get_macro_features(eosin_densities)
                
                # E. Morph & Spatial
                morph, spatial = get_morph_and_spatial(nuclei_mask)
                
                # 4. Concatenate
                feat_vec = np.concatenate([
                    hema_combined,  # 11
                    eosin_combined, # 7
                    [corr],         # 1
                    morph,          # 6
                    hema_macro,     # 2
                    eosin_macro,    # 2
                    spatial         # 4
                ])
                
                results.append((key_str, feat_vec))
    except Exception as e:
        print(f"Error in worker: {e}")
    finally:
        env.close()
        
    return results

# --- 3. Main Parallel Driver ---

def extract_features_parallel(
    ds_path, 
    output_filename, 
    num_workers=8, # Adjust based on your CPU cores
    limit=None
):
    w_matrix = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])
    
    # 1. Get all keys first (fast)
    logger.info("Reading keys from LMDB...")
    with lmdb.open(ds_path, readonly=True) as env:
        with env.begin() as txn:
            # Assuming you stored keys in a separate metadata entry or can iterate quickly
            # If iterating full DB is slow, use your existing metadata array if available
            cursor = txn.cursor()
            keys = []
            count = 0
            for k, _ in cursor:
                if b"_patch" in k:
                    keys.append(k.decode().split("_")[0])
                    count += 1
                    if limit and count >= limit: break
    
    logger.info(f"Found {len(keys)} patches. Starting parallel processing with {num_workers} workers.")
    
    # 2. Split keys into chunks
    chunk_size = len(keys) // num_workers
    chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
    
    # 3. Process
    all_features = []
    all_keys = []
    
    # Use 'spawn' for Windows compatibility with LMDB/Multiprocessing
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(num_workers) as pool:
        # Partial to pass constant arguments
        worker = partial(process_chunk, lmdb_path=ds_path, w_matrix=w_matrix)
        
        # Iterate over results as they finish
        for i, chunk_results in enumerate(pool.imap_unordered(worker, chunks)):
            if not chunk_results: continue
            
            # Unpack results
            batch_keys, batch_feats = zip(*chunk_results)
            all_keys.extend(batch_keys)
            all_features.extend(batch_feats)
            
            logger.info(f"Batch {i+1}/{len(chunks)} finished. Total processed: {len(all_features)}")

    # 4. Save
    logger.info("Saving results...")
    X = np.array(all_features)
    keys_arr = np.array(all_keys)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    np.savez(output_filename, features=X, keys=keys_arr)
    logger.success(f"Saved {len(X)} features to {output_filename}")

if __name__ == "__main__":
    # Ensure this protects the entry point for multiprocessing on Windows
    # python -m src.preproc_scripts.extract_features_paralle
    extract_features_parallel(
        ds_path="D:/tfm_data/preprocessed/lvl4_xgboost", 
        output_filename="c:/users/nicol/master_thesis/code/src/data/features_lvl4_xgboost.npz",
        num_workers=os.cpu_count() - 2 # Leave 2 cores free for OS
    )