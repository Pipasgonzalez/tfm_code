import numpy as np
import random
import math
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
# from albumentations.augmentations.geometric.functional import elastic_transform

from src.utils.augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from src.utils.augmenters.color.hedcoloraugmenter import HedColorAugmenter
from src.utils.augmenters.noise.gaussianbluraugmenter import GaussianBlurAugmenter
from src.utils.augmenters.noise.additiveguassiannoiseaugmenter import AdditiveGaussianNoiseAugmenter
from src.utils.augmenters.spatial.scalingaugmenter import ScalingAugmenter

# This is the "max level" from the paper
_MAX_LEVEL = 10.
_REPLACE_IMG = (128, 128, 128) # Default fill for images
_REPLACE_MASK = 0               # Default fill for masks

# --- Wrapper 1: For (H, W, C) augmenters (skimage, albumentations) ---
def _numpy_hwc_wrapper(func):
    def augmenter(image_pil, *func_args):
        # Convert PIL to Numpy. Shape is (H, W, C) or (H, W)
        img_np = np.asarray(image_pil) 
        
        # Apply the HWC function
        augmented_np = func(img_np, *func_args) 
        
        # Ensure type is correct before PIL conversion
        if augmented_np.dtype != np.uint8:
            augmented_np = np.clip(augmented_np, 0, 255).astype(np.uint8)

        return Image.fromarray(augmented_np)
    return augmenter

# --- Wrapper 2: For (C, H, W) augmenters (paper's custom spatial/noise) ---
def _numpy_cwh_wrapper(func):
    def augmenter(image_pil, *func_args):
        # Convert PIL to Numpy
        img_np = np.asarray(image_pil)
        
        # 1. Prepare for (C, H, W) format
        if img_np.ndim == 3: # (H, W, C)
            img_np = np.transpose(img_np, (2, 0, 1)) # -> (C, H, W)
            is_mask = False
        elif img_np.ndim == 2: # (H, W)
            img_np = np.expand_dims(img_np, axis=0) # -> (1, H, W)
            is_mask = True
        else:
            raise ValueError(f"Unexpected image dimension: {img_np.shape}")

        # Apply the (C, H, W) function
        augmented_np = func(img_np, *func_args)
        
        # 2. Convert back from (C, H, W) format
        if is_mask:
            augmented_np = np.squeeze(augmented_np, axis=0) # (1, H, W) -> (H, W)
            augmented_np = augmented_np.astype(np.uint8) 
            return Image.fromarray(augmented_np)
        else:
            augmented_np = np.transpose(augmented_np, (1, 2, 0)) # (C, H, W) -> (H, W, C)
            augmented_np = np.clip(augmented_np, 0, 255).astype(np.uint8)
            return Image.fromarray(augmented_np)
    return augmenter

# --- Scaling ---
# This function receives (C, H, W) and MUST return (C, H, W)
def _scaling_np(image, factor):
    if random.random() > 0.5:
        factor = factor / 60
        # Clamping max range to 1.2 to be reasonable
        augmentor = ScalingAugmenter(scaling_range=(1 - factor, 1.2), interpolation_order=1) 
    else:
        factor = factor / 30
        # Clamping max range to 1.5 to be reasonable
        augmentor = ScalingAugmenter(scaling_range=(1 + factor, 1.5), interpolation_order=1) 
    
    return augmentor.transform(image)
scaling = _numpy_cwh_wrapper(_scaling_np)

# --- HSV (HSB in paper's code) ---
# This function receives (C, H, W) and MUST return (C, H, W)
def _hsv_np(image, factor):
    if image.shape[0] != 3: # Don't apply color aug to mask
        return image
    factor = factor / 30
    if random.random() > 0.5:
        factor = -factor
    augmentor = HsbColorAugmenter(hue_sigma_range=factor, saturation_sigma_range=factor, brightness_sigma_range=factor)
    augmentor.randomize()
    return augmentor.transform(image)
hsv = _numpy_hwc_wrapper(_hsv_np)

# --- HED ---
# This function receives (C, H, W) and MUST return (C, H, W)
def _hed_np(image, factor):
    if image.shape[0] != 3: # Don't apply color aug to mask
        return image
    factor = factor / 30
    if random.random() > 0.5:
        factor = -factor
    augmentor = HedColorAugmenter(haematoxylin_sigma_range=factor, haematoxylin_bias_range=factor,
                                  eosin_sigma_range=factor, eosin_bias_range=factor,
                                  dab_sigma_range=factor, dab_bias_range=factor,
                                  cutoff_range=(0.15, 0.85))
    augmentor.randomize()
    return augmentor.transform(image)
hed = _numpy_hwc_wrapper(_hed_np) # Use HWC wrapper

# --- Gaussian Blur ---
# This function receives (C, H, W) and MUST return (C, H, W)
def _gauss_blur_np(image, factor):
    factor = factor / 5
    augmentor = GaussianBlurAugmenter(sigma_range=(factor, factor * 10))
    return augmentor.transform(image)
gauss_blur = _numpy_cwh_wrapper(_gauss_blur_np)

# --- Gaussian Noise ---
# This function receives (C, H, W) and MUST return (C, H, W)
def _gauss_noise_np(image, factor):
    factor = factor / 2
    augmentor = AdditiveGaussianNoiseAugmenter(sigma_range=(0.1 * factor, factor))
    return augmentor.transform(image)
gauss_noise = _numpy_cwh_wrapper(_gauss_noise_np)

def color(image, factor):
    factor = factor / 5 + 1
    return ImageEnhance.Color(image).enhance(factor)

def contrast(image, factor):
    factor = factor / 5 + 1
    return ImageEnhance.Contrast(image).enhance(factor)

def brightness(image, factor):
    factor = factor / 10 + 1
    return ImageEnhance.Brightness(image).enhance(factor)

def rotate(image, degrees, replace=_REPLACE_IMG):
    if random.random() > 0.5:
        degrees = -degrees
    return image.rotate(angle=degrees * 10, fillcolor=replace)

def translate_x(image, pixels, replace=_REPLACE_IMG):
    if random.random() > 0.5:
        pixels = -pixels
    pixels = pixels * 3
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=replace)

def translate_y(image, pixels, replace=_REPLACE_IMG):
    if random.random() > 0.5:
        pixels = -pixels
    pixels = pixels * 3
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=replace)

def shear_x(image, level, replace=_REPLACE_IMG):
    if random.random() > 0.5:
        level = -level
    level = level / 20
    return image.transform(image.size, Image.AFFINE, (1, level, 0, 0, 1, 0), Image.BICUBIC, fillcolor=replace)

def shear_y(image, level, replace=_REPLACE_IMG):
    if random.random() > 0.5:
        level = -level
    level = level / 20
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, level, 1, 0), Image.BICUBIC, fillcolor=replace)

def autocontrast(image):
    return ImageOps.autocontrast(image)

def identity(image):
    return image
  
def sharpness(image, factor):
    return ImageEnhance.Sharpness(image).enhance(factor)

def equalize(image):
    return ImageOps.equalize(image)

# --- 1. The "Translation Dictionary" (NAME_TO_FUNC) ---
# This maps a string name to the actual Python function
NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Hsv': hsv,
    'Hed': hed,
    'Identity': identity,
    'Equalize': equalize,
    'Rotate': rotate,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    # 'Elastic': elastic,
    'GaussBlur': gauss_blur,
    'GaussNoise': gauss_noise,
    'Scaling': scaling
}

def _randomly_negate_tensor(tensor):
    return -tensor if random.random() > 0.5 else tensor

# --- 2. The "Argument Translator" (level_to_arg) ---
# This translates the single 'level' (0-M) into the 
# specific argument the function needs.
def _level_to_arg(level):
    return (level,)

def level_to_arg(hparams=None): # hparams is unused, just for compatibility
    return {
        'Identity': lambda level: (),
        'Hsv': _level_to_arg,
        'Hed': _level_to_arg,
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Rotate': _level_to_arg,
        'Color': _level_to_arg,
        'Contrast': _level_to_arg,
        'Brightness': _level_to_arg,
        'Sharpness': _level_to_arg,
        'ShearX': _level_to_arg,
        'ShearY': _level_to_arg,
        'TranslateX': _level_to_arg,
        'TranslateY': _level_to_arg,
        # 'Elastic': _level_to_arg,
        'GaussBlur': _level_to_arg,
        'GaussNoise': _level_to_arg,
        'Scaling': _level_to_arg,
    }