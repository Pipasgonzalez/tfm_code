# from tiatoolbox.tools.stainnorm import get_normalizer
import tiatoolbox
import openslide
from tiatoolbox.wsicore import WSIReader
import numpy as np
from tiatoolbox.tools.stainnorm import MacenkoNormalizer

# from tiatoolbox.utils.transforms import od2rgb, rgb2od


class RobustMacenkoNormalizer(MacenkoNormalizer):
    """
    A robust version of the MacenkoNormalizer.

    This class overrides the transform method. It first attempts the standard
    Macenko transformation. If it fails (usually due to an empty tissue mask
    in a pale patch), it implements a fallback strategy: it uses the
    pre-computed target stain matrix to calculate the stain concentrations
    of the source patch and then normalizes.

    This prevents data loss from skipping patches that are difficult to normalize.
    """

    def __init__(self):
        super().__init__()
        self.total_counter = 0
        self.exception_counter = 0

    def transform(self, img: np.ndarray) -> np.ndarray:
        self.total_counter += 1
        try:
            # First, try the original method, which is best for most patches
            return super().transform(img)
        except ValueError:
            # If the above fails, it's likely due to "Empty tissue mask".
            # We now implement our fallback logic.

            # Use the STABLE stain matrix from the target image (`fit` step)
            # This is the key to our fallback: we assume the problematic
            # patch has a stain profile close to our ideal target.
            stain_matrix_source = self.stain_matrix_target

            # Get concentrations of the source image using the fallback stain matrix
            source_concentrations = self.get_concentrations(img, stain_matrix_source)

            # Normalize the concentrations as before
            max_c_source = np.percentile(source_concentrations, 99, axis=0).reshape(
                (1, 2)
            )

            # Avoid division by zero if max_c_source is 0
            # This can happen on patches with no stain at all
            if np.any(max_c_source == 0):
                return img  # Return the original image if no stain is detected

            source_concentrations *= self.maxC_target / max_c_source

            # Reconstruct the image
            trans = 255 * np.exp(
                -1 * np.dot(source_concentrations, self.stain_matrix_target)
            )

            # Ensure values are within the 0-255 range
            trans[trans > 255] = 255
            trans[trans < 0] = 0

            self.exception_counter += 1
            return trans.reshape(img.shape).astype(np.uint8)


def get_wsi_normalizer(
    target_size: int = 2048,
    target_level: int = 2,
    target_fp: str = "D:/tfm_data/labeled_wsi/H15-16121-IA.mrxs",
):
    """
    Fits a normalizer on a preselected reference WSI and returns it already transformed. We use the Macenko method atm
    """
    target_wsi_os = openslide.OpenSlide(filename=target_fp)
    start_x = int(target_wsi_os.properties["openslide.bounds-x"])
    start_y = int(target_wsi_os.properties["openslide.bounds-y"])
    tia_wsi = WSIReader.open(target_fp)

    ds_levels = target_wsi_os.level_downsamples
    ds_ratio = int(ds_levels[target_level])
    # Get mpp
    mpp_x, mpp_y = (
        target_wsi_os.properties["openslide.mpp-x"],
        target_wsi_os.properties["openslide.mpp-y"],
    )
    roi_x_rel, roi_y_rel = 8340, 11340
    scaled_roi_x, scaled_roi_y = int(roi_x_rel / float(mpp_x)), int(
        roi_y_rel / float(mpp_y)
    )  # Level 0 Pixel Locations
    start_roi_x, start_roi_y = start_x + scaled_roi_x, start_y + scaled_roi_y

    roi = tia_wsi.read_bounds(
        bounds=(
            start_roi_x,
            start_roi_y,
            start_roi_x + target_size * ds_ratio,
            start_roi_y + target_size * ds_ratio,
        ),
        resolution=target_level,
        units="level",
    )
    # target_image=Image.fromarray(roi)
    normalizer = RobustMacenkoNormalizer()
    normalizer.fit(roi)
    return normalizer


def translate_to_tissue_coords(
    x, y, start_x_ds, start_y_ds, scale_x, scale_y, padding_amount
):
    x_relative_mask = x - start_x_ds
    y_relative_mask = y - start_y_ds
    scaled_x_mask = x_relative_mask * scale_x
    scaled_y_mask = y_relative_mask * scale_y
    x_mask = int(scaled_x_mask) + padding_amount
    y_mask = int(scaled_y_mask) + padding_amount
    return x_mask, y_mask
