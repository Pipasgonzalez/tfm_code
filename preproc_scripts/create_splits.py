import pandas as pd
from pathlib import Path
import lmdb
import numpy as np
import io
import random
import matplotlib.pyplot as plt
import json
import argparse 
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--dhgp_csv_path", type=str, default = "C:/users/nicol/master_thesis/wsi/dhgp_scores.csv")

def main(dhgp_csv_path, num_splits = 3, labeled_per_category = 4, test_per_category = 1):
    table = pd.read_csv(dhgp_csv_path, encoding="latin-1", sep=";")
    env = lmdb.open("D:/tfm_data/preprocessed/lmdb_raw",readonly=True)
    with env.begin() as txn:
        metadata_buffer = txn.get(b"__metadata")
        if metadata_buffer is None:
            raise ValueError("No metadata found in LMDB")
        metadata_array = np.load(io.BytesIO(metadata_buffer), allow_pickle=True)
        metadata = list(zip(
        [key.decode('utf-8') for key in metadata_array["keys"]], # <--- Decoding step
        metadata_array["wsi_ids"]
            ))
        wsi_ids_in_lmdb = set([wsi_id for _, wsi_id in metadata])

    is_in_lmdb = table['EMC_flnm'].isin(wsi_ids_in_lmdb)
    annotated_rows = table[is_in_lmdb].copy()
    annotated_rows['dHGP'] = pd.to_numeric(annotated_rows['dHGP'], errors='coerce')
    annotated_rows.dropna(subset=['dHGP'], inplace=True) # Drop rows where dHGP is not a number


    # 1. Define constants for clarity
    total_samples_per_category = labeled_per_category + test_per_category # This will be 5

    # 2. Categorize the WSIs based on dHGP score
    df_zero = annotated_rows[annotated_rows['dHGP'] == 0]
    df_mid = annotated_rows[(annotated_rows['dHGP'] >= 1) & (annotated_rows['dHGP'] <= 99)]
    df_hundred = annotated_rows[annotated_rows['dHGP'] == 100]

    # 3. Check if you have enough data for all splits BEFORE starting
    required_samples = num_splits * total_samples_per_category # 5 splits * 3 samples = 15
    if len(df_zero) < required_samples or len(df_mid) < required_samples or len(df_hundred) < required_samples:
        print("\n--- ERROR: Not enough data for the requested number of splits. ---")
        print(f"Required unique samples from each category: {required_samples}")
        print(f"Available 'Zero' samples: {len(df_zero)}")
        print(f"Available 'Mid' samples: {len(df_mid)}")
        print(f"Available 'Hundred' samples: {len(df_hundred)}")
        # Decide here if you want to exit or proceed with fewer splits
        # For now, we will proceed and the loop will break automatically.

    # 4. Create shuffled lists of WSI IDs for random sampling
    ids_zero = df_zero['EMC_flnm'].tolist()
    ids_mid = df_mid['EMC_flnm'].tolist()
    ids_hundred = df_hundred['EMC_flnm'].tolist()
    random.shuffle(ids_zero)
    random.shuffle(ids_mid)
    random.shuffle(ids_hundred)

    # 5. Generate the splits
    all_splits = {}
    total_wsi_ids = set(annotated_rows['EMC_flnm'])

    for i in range(num_splits):
        split_name = f"split_{i+1}"
        
        # Calculate the start and end indices for slicing the lists
        start_idx = i * total_samples_per_category
        end_idx = (i + 1) * total_samples_per_category
        
        # Check if there are enough unique samples remaining for the current split
        if end_idx > len(ids_zero) or end_idx > len(ids_mid) or end_idx > len(ids_hundred):
            print(f"\nStopping at {split_name}: Not enough unique samples left for a full split.")
            break
            
        # Sample 3 IDs from each category for the current split
        samples_zero = ids_zero[start_idx:end_idx]
        samples_mid = ids_mid[start_idx:end_idx]
        samples_hundred = ids_hundred[start_idx:end_idx]
        
        # --- THIS IS THE CORE LOGIC CHANGE ---
        # Assign the first 2 of each sample group to the 'labeled' set
        labeled_ids = (samples_zero[:labeled_per_category] + 
                    samples_mid[:labeled_per_category] + 
                    samples_hundred[:labeled_per_category])
        
        # Assign the last 1 of each sample group to the 'test' set
        test_ids = (samples_zero[labeled_per_category:] + 
                    samples_mid[labeled_per_category:] + 
                    samples_hundred[labeled_per_category:])
        
        # The 'unlabeled' set is everything else
        labeled_and_test_ids = set(labeled_ids + test_ids)
        unlabeled_ids = list(total_wsi_ids - labeled_and_test_ids)
        
        # Store the results for the current split
        all_splits[split_name] = {
            "labeled_wsi_ids": labeled_ids,
            "test_wsi_ids": test_ids,
            "unlabeled_wsi_ids": unlabeled_ids
        }
        print(f"Created {split_name} with {len(labeled_ids)} labeled, {len(test_ids)} test, and {len(unlabeled_ids)} unlabeled WSIs.")

    # 6. Save the splits to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"C:/users/nicol/master_thesis/code/src/config/splits_{timestamp}.json"
    output_path = Path(output_filename) # Directly use the full path

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_splits, f, indent=4)

    print(f"\nSuccessfully saved {len(all_splits)} splits to '{output_path}'")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dhgp_csv_path)