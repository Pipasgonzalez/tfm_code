# from tiatoolbox.annotation import Annotation
from shapely.geometry import shape, mapping, MultiPolygon
import json
from shapely.affinity import translate
import numpy as np
from copy import deepcopy
from PIL import Image

# Edited Merged Classes (4)
# CLASS_MAPPING = {
#     "background": 0,
#     "liver_portal_triad_border_effective": 1,  # tumor/liver border --> merge with border
#     "liver_portal_triad_nborder_effective": 2,  # --> merge with liver
#     "tumor_mucus_effective": 3,  # merge with 4
#     "tumor_cells_effective": 3,
#     "desmoplastic_ring_effective": 1,  # alone
#     "liver_normal_effective": 2,
#     # New Ones
#     "liver_normal_capsule_effective":2, # Connective tissue that is on the border and is liver but looks like dHGP
#     "tumor_necrosis_effective": 3
# }

CLASS_MAPPING = {
    "background": 0,
    "liver_portal_triad_border_effective": 1,
    "liver_portal_triad_nborder_effective": 2,
    "tumor_mucus_effective": 3,  # merge with 4
    "tumor_cells_effective": 3,
    "desmoplastic_ring_effective": 1,  # alone
    "liver_normal_effective": 2,
    # New Ones
    "liver_normal_capsule_effective":2, # Connective tissue that is on the border and is liver but looks like dHGP
    "tumor_necrosis_effective": 3
}
CLASS_MAPPING_XGBOOST = {
    "background": 0,
    "liver_portal_triad_border_effective": 1,
    "liver_portal_triad_nborder_effective": 1,
    "tumor_mucus_effective": 2,
    "tumor_cells_effective": 3,
    "desmoplastic_ring_effective": 4,  # alone
    "liver_normal_effective": 5,
    "liver_normal_capsule_effective":5, # Connective tissue that is on the border and is liver but looks like dHGP
    "tumor_necrosis_effective": 6
}


CLASS_COLOR = {
    "background": (0, 0, 0, 255),
    "liver_portal_triad_border_effective": (50, 50, 50, 255),
    "liver_portal_triad_nborder_effective": (75, 75, 75, 255),
    "tumor_mucus_effective": (100, 100, 100, 255),
    "tumor_cells_effective": (150, 150, 150, 255),
    "desmoplastic_ring_effective": (200, 200, 200, 255),
    "liver_normal_effective": (250, 250, 250, 255),

    # NEW ONES
    "liver_normal_capsule_effective":(25,25,25,255), # Connective tissue that is on the border and is liver but looks like dHGP
    "tumor_necrosis_effective": (225,225,225,255)
}
CLASS_ID_TO_COLOR = {
    id: color for id, color in zip(CLASS_MAPPING.values(), CLASS_COLOR.values())
}


# No changes needed to color_mapper function itself
def color_mapper(ann_props: dict):
    label = ann_props["type"]
    # .get will now return a 4-element tuple
    return CLASS_COLOR.get(label, (0, 0, 0, 255))  # Default to opaque black


# def unpack_qupath_classification(ann: Annotation) -> Annotation:
#     """
#     Helper function to unpack the nested 'classification' dictionary
#     from a QuPath GeoJSON export.
#     """
#     props = ann.properties
#     # Check if the 'classification' key exists
#     if "classification" in props:
#         # Get the nested dictionary and remove it from its original location
#         classification_data = props.pop(
#             "classification"
#         )  # Add the 'name' and 'color' to the top-level properties
#         if "name" in classification_data:
#             props["type"] = classification_data["name"]
#         if "color" in classification_data:
#             props["color"] = classification_data["color"]
#     return ann


def transform_properties(props: dict):
    new_props = {
        "objectType": "annotation",
        "type": props["classification"]["name"],
        "color": CLASS_COLOR.get(props["classification"]["name"], (0, 0, 0, 255)),
    }
    return new_props


def preprocess_geojson_split_multipolygon(
    input_path, output_path, offset_x=0, offset_y=0
):
    """
    Translates all geometries by a given offset and converts any
    MultiPolygon features into individual Polygon features.
    """
    with open(input_path) as f:
        data = json.load(f)

    new_features = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        translated_geom = translate(geom, xoff=offset_x, yoff=offset_y)

        # Transform properties for ALL features
        transformed_props = transform_properties(feature["properties"])

        if isinstance(translated_geom, MultiPolygon):
            for poly in translated_geom.geoms:
                new_features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(poly),
                        "properties": transformed_props,
                    }
                )
        else:
            # If it's not a MultiPolygon, update the feature's geometry
            # and use the transformed properties
            feature["geometry"] = mapping(translated_geom)
            feature["properties"] = transformed_props
            new_features.append(feature)

    data["features"] = new_features

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# THIS WAS SEVERELY WRONG
# def get_annotation_imgs(annotation_patch: np.ndarray):
#     anno_img = Image.fromarray(annotation_patch)
#     anno_img_gray = anno_img.convert("L")

#     h, w, _ = annotation_patch.shape
#     class_mask = np.zeros((h, w), dtype=np.uint8)
#     for class_id, color in CLASS_ID_TO_COLOR.items():
#         match = np.all(annotation_patch == color[:3], axis=-1)
#         class_mask[match] = class_id
#     mask_img = Image.fromarray(class_mask)
#     return anno_img_gray, mask_img, class_mask


# FIX: Rewrite get_annotation_imgs
def get_annotation_imgs(annotation_patch: np.ndarray):
    anno_img = Image.fromarray(annotation_patch)
    anno_img_gray = anno_img.convert("L")
    h, w, _ = annotation_patch.shape
    class_mask = np.zeros((h, w), dtype=np.uint8) # This is good
    # Iterate over the original string-based color map
    for class_name, color in CLASS_COLOR.items():
    # Find the matching pixels
        match = np.all(annotation_patch == color[:3], axis=-1)
    

        class_id = CLASS_MAPPING.get(class_name) 
    
        # Only assign if the class_id is valid (not None)
        if class_id is not None:
            class_mask[match] = class_id
        
    mask_img = Image.fromarray(class_mask)
    return anno_img_gray, mask_img, class_mask

import torchvision.utils
import torch
import os
from PIL import Image
def save_batch_images(batch_patches, batch_targets, output_dir, batch_idx, num_classes=4):
    """
    Saves a batch of patches and their corresponding color-mapped targets.
    Iterates over the batch and saves each image individually using PIL.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Convert Tensors to CPU NumPy Arrays ---
    
    # Patches: [B, C, H, W], float, [0, 1]
    # Move to CPU, convert to NumPy, and transpose to [B, H, W, C]
    patches_np = batch_patches.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Targets: [B, H, W], long
    # Move to CPU and convert to NumPy
    targets_np = batch_targets.cpu().numpy().astype(np.uint8)

    # --- 2. Define NumPy Color Map ---
    color_map_np = np.array([
        [0, 0, 0],       # 0: Background (Black)
        [255, 0, 0],     # 1: Class 1 (Red)
        [0, 255, 0],     # 2: Class 2 (Green)
        [0, 0, 255]      # 3: Class 3 (Blue)
    ], dtype=np.uint8)

    # --- 3. Iterate Over the Batch and Save ---
    batch_size = patches_np.shape[0]
    for i in range(batch_size):
        
        # --- Process and Save Patch ---
        
        # Get the i-th patch (H, W, C)
        patch_image_np = patches_np[i]
        
        # Scale from [0, 1] float to [0, 255] uint8
        patch_image_uint8 = (patch_image_np * 255.0).astype(np.uint8)
        
        # Create PIL Image from (H, W, 3) NumPy array
        patch_pil = Image.fromarray(patch_image_uint8, 'RGB')
        
        # Save the image
        patch_pil.save(os.path.join(output_dir, f"batch_{batch_idx}_idx_{i}_patch.png"))

        # --- Process and Save Target ---
        
        # Get the i-th target (H, W)
        target_mask_np = targets_np[i]
        
        # Map class indices to RGB colors -> (H, W, 3)
        target_image_uint8 = color_map_np[target_mask_np]
        
        # Create PIL Image from (H, W, 3) NumPy array
        target_pil = Image.fromarray(target_image_uint8, 'RGB')
        
        # Save the image
        target_pil.save(os.path.join(output_dir, f"batch_{batch_idx}_idx_{i}_target.png"))


# Define the color map as a global variable
COLOR_MAP_NP = np.array([
    [0, 0, 0],      # 0: Background (Black)
    [255, 0, 0],    # 1: Class 1 (Red)
    [0, 255, 0],    # 2: Class 2 (Green)
    [0, 0, 255]     # 3: Class 3 (Blue)
], dtype=np.uint8)

def map_seg_to_color_np(seg_np, color_map):
    """
    Maps a 2D segmentation numpy array (H, W) to a 3-channel color image (3, H, W).
    """
    # seg_np is (H, W), dtype int
    color_image = color_map[seg_np.astype(np.int64)]
    # Transpose from (H, W, 3) to (3, H, W) for TensorBoard
    return np.transpose(color_image, (2, 0, 1))

def normalize_patch_np(patch_np):
    """
    Converts a (C, H, W) float numpy array in [0, 1] 
    to a (C, H, W) or (1, H, W) uint8 array in [0, 255].
    """
    # 1. Scale from [0.0, 1.0] to [0.0, 255.0]
    scaled_np = patch_np * 255.0
    
    # 2. Clip to ensure it's in range (in case of small float errors)
    #    and cast to uint8
    scaled_8bit = np.clip(scaled_np, 0, 255).astype(np.uint8)
    
    # 3. Handle channel dimension for TensorBoard
    if scaled_8bit.ndim == 2:
        # (H, W) -> (1, H, W) for grayscale
        return np.expand_dims(scaled_8bit, axis=0)
    elif scaled_8bit.shape[0] == 1 or scaled_8bit.shape[0] == 3:
        # (1, H, W) or (3, H, W) is already good
        return scaled_8bit
    else:
        # (C, H, W) with C != 1 or 3. 
        # Your image is RGB (3 channels), so this should be fine.
        # If it were > 3, we'd have a problem.
        return scaled_8bit
    



def log_batch_to_tensorboard(batch_data, prefix, epoch, writer):
    """
    Logs the FIRST patch from a saved batch to TensorBoard.
    """
    loss, img_batch_np, tgt_batch_np, pred_batch_np = batch_data
    
    # --- Select the FIRST patch (index 0) from the saved batch ---
    image_patch_np = img_batch_np[0] 
    target_patch_np = tgt_batch_np[0]
    pred_patch_np = pred_batch_np[0]
    # ---
    
    # Handle 3D or 2D (copied from your code)
    is_3d = image_patch_np.ndim == 4
    if is_3d:
        mid_slice_idx = image_patch_np.shape[1] // 2
        image_to_log_np = image_patch_np[:, mid_slice_idx, :, :]
        target_to_log_np = target_patch_np[mid_slice_idx, :, :]
        pred_to_log_np = pred_patch_np[mid_slice_idx, :, :]
    else:
        image_to_log_np = image_patch_np
        target_to_log_np = target_patch_np
        pred_to_log_np = pred_patch_np
    
    # 1. Normalize original image (NumPy)
    img_normalized = normalize_patch_np(image_to_log_np)
    
    # 2. Color-map ground truth (NumPy)
    target_color = map_seg_to_color_np(target_to_log_np, COLOR_MAP_NP)
    
    # 3. Color-map prediction (NumPy)
    pred_color = map_seg_to_color_np(pred_to_log_np, COLOR_MAP_NP)

    # 4. Log to TensorBoard (writer accepts NumPy arrays)
    # Add the batch-average loss to the tag
    tag_prefix = f"Validation/{prefix}_(BatchLoss_{loss:.4f})" 
    writer.add_image(f"{tag_prefix}_Image", img_normalized, epoch, dataformats='CHW')
    writer.add_image(f"{tag_prefix}_GroundTruth", target_color, epoch, dataformats='CHW')
    writer.add_image(f"{tag_prefix}_Prediction", pred_color, epoch, dataformats='CHW')


def map_predictions_to_ground_truth_dataset(
    pred_meta_path="detailed_dataset_metadata.npz",
    pred_values_path="entropy_predictions.npy", # Your XGBoost outputs
    gt_meta_path="training_dataset_metadata.npz"
):
    # 1. Load Prediction Data (Detailed)
    pred_meta = np.load(pred_meta_path, allow_pickle=True)
    pred_entropies = np.load(pred_values_path)
    
    # Create a lookup dictionary: Key -> Entropy
    # Key = f"{wsi_id}_{x}_{y}"
    print("Building Prediction Lookup Table...")
    pred_lookup = {}
    for i in range(len(pred_meta['keys'])):
        w_id = pred_meta['wsi_ids'][i]
        x = pred_meta['coords_x'][i]
        y = pred_meta['coords_y'][i]
        
        unique_key = (w_id, x, y)
        pred_lookup[unique_key] = pred_entropies[i]

    # 2. Load Ground Truth Training Data (Simple)
    gt_meta = np.load(gt_meta_path, allow_pickle=True)
    
    # 3. Align
    print("Mapping predictions to Training Dataset...")
    aligned_entropies = []
    found_count = 0
    missing_count = 0
    
    for i in range(len(gt_meta['keys'])):
        w_id = gt_meta['wsi_ids'][i]
        x = gt_meta['coords_x'][i]
        y = gt_meta['coords_y'][i]
        
        unique_key = (w_id, x, y)
        
        if unique_key in pred_lookup:
            # We found the exact same patch in the prediction set
            aligned_entropies.append(pred_lookup[unique_key])
            found_count += 1
        else:
            # Should not happen if datasets are identical, 
            # but good for safety (e.g. slight difference in tissue masking?)
            aligned_entropies.append(-1) 
            missing_count += 1
            
    print(f"Mapped {found_count} patches. Missing {missing_count}.")
    
    return np.array(aligned_entropies)