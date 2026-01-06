import cv2
import numpy as np
from histomicstk.preprocessing.color_deconvolution import color_deconvolution
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

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

def get_all_features(patch_np, w=[[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]]):
    h_e = color_deconvolution(patch_np, w)
    hema = h_e.Stains[:, :, 0].astype(np.uint8)
    eosin = h_e.Stains[:, :, 1].astype(np.uint8)

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
    return features