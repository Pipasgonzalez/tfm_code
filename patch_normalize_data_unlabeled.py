import os
import numexpr
import torch

# Setup Logger
from loguru import logger
import sys
import json

logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# To remove annoying info logs
os.environ["NUMEXPR_LOG_LEVEL"] = "WARN"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
numexpr.set_num_threads(1)
torch.set_num_threads(1)

from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image
from collections import defaultdict
import argparse
import time
import traceback
import multiprocessing as mp

OPENSLIDE_PATH = r"C:\Users\nicol\master_thesis\code\openslide-bin-4.0.0.8-windows-x64\openslide-bin-4.0.0.8-windows-x64\bin"
# if hasattr(os, "add_dll_directory"):
#     os.add_dll_directory(OPENSLIDE_PATH)
import openslide

from tiatoolbox.wsicore import WSIReader
from src.utils.normalizer import get_wsi_normalizer
from src.utils.wsi import translate_to_tissue_coords
from src.preproc_scripts.lmdb_dataset import UnlabeledDatasetLMDB

parser = argparse.ArgumentParser()
parser.add_argument("--wsi_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--res_level", type=int, default=4)
parser.add_argument("--patch_size", type=int, default=512)
parser.add_argument("--stride", type=int, default=512)
parser.add_argument("--min_tissue", type=float, default=0.25)
args = parser.parse_args()


def writer_process(lmdb_path, queue, num_files,resolution):
    db = UnlabeledDatasetLMDB(lmdb_path, num_wsi=num_files, resolution=resolution)
    none_count = 0
    while none_count < num_files:
        item = queue.get()
        if item is None:
            none_count += 1
            logger.info(f"Currently at {none_count} processed WSI IDs")
            continue
        patch, wsi_id, coords = item
        res = db.save_patch(
            patch=patch, wsi_id=wsi_id, coords = coords
        ) 
        if not res:
            print("Warning: Failed to save patch in writer")
    db.close()
    print("Successfully saved all patches and metadata to DB")


def preprocess_image(
    wsi_path, res_level, patch_size, stride, min_tissue_ratio, queue
):
    try:
        stem = Path(wsi_path).stem
        logger.info(f"Starting Processing {stem}")

        # Objects used for this WSI
        normalizer = get_wsi_normalizer(get_prefitted=True, device="cpu")

        # Open WSI
        tia_wsi = WSIReader.open(wsi_path)
        wsi = openslide.OpenSlide(wsi_path)
        
        logger.info("Opened Whole Slides")

        # Get offsets
        start_x = int(wsi.properties["openslide.bounds-x"])
        start_y = int(wsi.properties["openslide.bounds-y"])

        # Load tissue mask (adjust path if needed)
        tissue_mask_img = Image.open("D:/tfm_data/binary_masks_unlabeled_all/" + stem + ".png")
        tissue_mask_unpadded = np.array(tissue_mask_img)

        logger.info("Opened Tissue Segmentation")

        # Calculate scales
        samp_lvl = wsi.level_downsamples[res_level]
        mpp_x, mpp_y = float(wsi.properties["openslide.mpp-x"]), float(
            wsi.properties["openslide.mpp-y"]
        )
        mpp_res_lvl_x, mpp_res_lvl_y = samp_lvl * mpp_x, samp_lvl * mpp_y
        scale_x, scale_y = mpp_res_lvl_x / 10, mpp_res_lvl_y / 10

        patch_size_mask_x = int(patch_size * scale_x)
        patch_size_mask_y = int(patch_size * scale_y)

        padding_amount = max(patch_size_mask_x, patch_size_mask_y)
        tissue_mask = np.pad(
            tissue_mask_unpadded,
            pad_width=padding_amount,
            mode="constant",
            constant_values=0,
        )

        # Bounds
        width = int(wsi.properties["openslide.bounds-width"])
        height = int(wsi.properties["openslide.bounds-height"])
        end_x = start_x + width
        end_y = start_y + height

        downsample_level = tia_wsi.info.level_downsamples[res_level]
        start_x_ds = int(start_x / downsample_level)
        start_y_ds = int(start_y / downsample_level)
        end_x_ds = int(end_x / downsample_level)
        end_y_ds = int(end_y / downsample_level)

        logger.info("Extracted Infos from mrxs file")

        # Extract patches
        extracted_patch_count = 0
        total_patch_count = 0
        background_filtered_patch_count = 0
        failed_patch_count = 0
        for y in range(start_y_ds, end_y_ds, stride):
            for x in range(start_x_ds, end_x_ds, stride):
                total_patch_count += 1
                x_mask, y_mask = translate_to_tissue_coords(
                    x, y, start_x_ds, start_y_ds, scale_x, scale_y, padding_amount
                )

                # Safety out of bounds measure. Shouldn't happen though
                if (
                    y_mask + patch_size_mask_y > tissue_mask.shape[0]
                    or x_mask + patch_size_mask_x > tissue_mask.shape[1]
                ):
                    logger.warning(
                        f"Out of Bounds Error Avoided for Tissue Mask of ID: {str(stem)}"
                    )
                    failed_patch_count += 1
                    continue

                mask_region = tissue_mask[
                    y_mask : y_mask + patch_size_mask_y,
                    x_mask : x_mask + patch_size_mask_x,
                ]

                # If the patch of the tissue mask consists of zero pixels, continue. Shouldn't happen, but just in case
                if mask_region.size == 0:
                    failed_patch_count += 1
                    logger.warning(
                        f"Mask Image with Size 0 detected for WSI {str(stem)}"
                    )
                    continue

                # Calculate the percentage tissue amount of the slide
                tissue_ratio = np.mean(mask_region) / 255
                if tissue_ratio < min_tissue_ratio:
                    background_filtered_patch_count += 1
                    continue

                location_level0 = (int(x * downsample_level), int(y * downsample_level))
                patch = tia_wsi.read_rect(
                    location=location_level0,
                    size=(patch_size, patch_size),
                    resolution=res_level,
                    units="level",
                )
                try:
                    normalized_patch, _ , _ = normalizer.normalize(patch)
                    normalized_patch_np = normalized_patch.cpu().permute(1, 2, 0).numpy()
                    normalized_patch_cpu = np.clip(normalized_patch_np, 0, 255).astype(np.uint8) # In order to avoid wrap around
                except Exception as e:
                    logger.info(f"Exception during normalizing: {str(e)}")
                    failed_patch_count += 1
                    continue

                if normalized_patch is not None:
                    queue.put(
                        (
                            normalized_patch_cpu,
                            str(Path(wsi_path).stem),
                            location_level0
                        )
                    )  # Send to queue
                    extracted_patch_count += 1
        
        queue.put(None)  # Sentinel for this worker
        print("Total Number of Successful Normalizations", normalizer.success_counter, "Number of Exceptions", normalizer.exception_counter)
        return_dict = {
            "wsi_id": str(stem),
            "total_processed": total_patch_count,
            "extracted": extracted_patch_count,
            "background_filtered": background_filtered_patch_count,
            "failed": failed_patch_count,
        }
        logger.info(str(return_dict))
        return return_dict

    except Exception as e:
        print(f"Error processing {wsi_path}: {e}")
        traceback.print_exc()
        queue.put(None)
        return 0


def preprocess_images_parallel(
    wsi_dir: str,
    output_dir: str,
    res_level: int,
    patch_size: int,
    stride: int,
    min_tissue_ratio: float,
    remote:bool = False
):
    if not all(os.path.isdir(d) for d in [wsi_dir, output_dir]):
        raise ValueError("One or more directories are invalid.")

    # Group files by stem
    if remote:
        with open("./src/config/all_unlabeled_wsi_ids.json") as f:
            unlabeled_wsi_ids = json.load(f)
            
        all_unlabeleled_wsi_ids = unlabeled_wsi_ids["all_unlabeled_wsi_ids"]
        labeled_wsi_ids = ["H18-24578-IIB",
            "H15-15551-IIB",
            "D00-41129-IIIC",
            "H18-16958-IIA",
            "D01-41285-IIIB",
            "H18-15067-A",
            "H01-9856-C",
            "H17-4726-IIA",
            "H16-5021-IIC",
            "H18-9736-B",
            "H16-5021-IIA",
            "D02-40504-IIA",
            "H16-12643-IA",
            "H18-12073-IIA",
            "D09-42190-E",
            "D09-42190-A",
            "H18-20317-IIA",
            "H15-16121-IA",
            "H16-17142-II",
            "D10-41545-A",
            "H14-12726-IIA",
            "H18-10906-IIIA",
            "H15-7933-IVC",
            "H18-16599-IA",
            "H18-24578-IIA",
            "H18-11547-IIA",
            "H18-20690-A",
            "H16-23157-IIA",
            "H16-12870-IIB",
            "D02-40593-IIIA",
            "D02-40504-IIIB",
            "D00-40067-IIA",
            "D00-41129-IIIA",
            "H18-23553-IA",
            "H18-12073-IIIB",
            "H18-2220-IA",
            "D00-42686-IIIA",
            "D01-43315-IIIA",
            "H18-4312-IVA",
            "H18-14189-IIB"]
        set_all_unlabeled = set(all_unlabeleled_wsi_ids)
        set_all_labeled = set([id+".mrxs" for id in labeled_wsi_ids])
        filtered = [item for item in set_all_unlabeled if item not in set_all_labeled]
        print("Length of all unlabeled", len(set_all_unlabeled), "Length after filtering", len(filtered))
        mrxs_files = [Path(os.path.join(wsi_dir,wsi_id)) for wsi_id in filtered]
    else:
        print("Yes")
        mrxs_files = list(Path(wsi_dir).glob("*.mrxs"))

    # Prepare inputs: list of (wsi_path, anno_path)
    inputs = []
    for wsi_file in mrxs_files:
        inputs.append(
            (
                str(wsi_file),
                res_level,
                patch_size,
                stride,
                min_tissue_ratio,
            )
        )

    num_files = len(inputs)
    print(f"Found {num_files} WSI files to process")

    num_workers = 2
    print(f"Using {num_workers} worker processes")

    manager = mp.Manager()
    queue = manager.Queue(maxsize=10000)

    # Start writer process
    writer_p = mp.Process(target=writer_process, args=(output_dir, queue, num_files,res_level))
    writer_p.daemon = True  #
    writer_p.start()

    # Add queue to inputs
    inputs_with_queue = [inp + (queue,) for inp in inputs]

    # Process in parallel
    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(preprocess_image, inputs_with_queue)

        # Wait for writer to finish
        writer_p.join()
        try:
            # Aggregate results into a DataFrame
            results_df = pd.DataFrame(results)
            # Save to CSV with current datetime
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("./src/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            output_csv = f"./src/logs/PatchExtraction_{current_datetime}.csv"
            results_df.to_csv(output_csv, index=False)
            print(f"Saved extraction stats to {output_csv}")
        except Exception:
            logger.error("Failed Saving Results to CSV")
            print(results)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        writer_p.terminate()
        writer_p.join()
        raise


if __name__ == "__main__":
    args.stride = args.stride or (args.patch_size // 2)

    # Downsampled version
    # python -m src.patch_normalize_data_unlabeled --wsi_dir D:/tfm_data/unlabeled_wsi_new --output_dir D:/tfm_data/preprocessed/dataset_unlabeled_lvl4 --res_level 4
    
    # To Process on the Mijn Werkplek
    # output_dir = r"\\store\department\heelkunde\coupes\Annoteren_Zhen_Qian\nico_data\lvl4_unlabeled_all"
    # wsi_dir = r"\\store\department\heelkunde\coupes\Erasmus_MC"
    output_dir = "D:/tfm_data/preprocessed/dataset_unlabeled_lvl4"
    wsi_dir = "D:/tfm_data/unlabeled_wsi_new"
    start = time.time()
    preprocess_images_parallel(
        res_level=args.res_level,
        patch_size=args.patch_size,
        min_tissue_ratio=args.min_tissue,
        wsi_dir=wsi_dir,
        output_dir=output_dir,
        stride=args.stride,
        remote=False
    )
    logger.info(f"Finished Extracting Patches in {(time.time()-start)/60} minutes")


# Total Number of Successful Normalizations 0 Number of Exceptions 0
# 2025-11-30T21:12:48.739575+0100 INFO {'wsi_id': 'H16-1975-IIA', 'total_processed': 180, 'extracted': 0, 'background_filtered': 78, 'failed': 102}
# Total Number of Successful Normalizations 0 Number of Exceptions 0
# 2025-11-30T21:12:48.776565+0100 INFO {'wsi_id': 'D12-40966-IIA', 'total_processed': 195, 'extracted': 0, 'background_filtered': 78, 'failed': 117}