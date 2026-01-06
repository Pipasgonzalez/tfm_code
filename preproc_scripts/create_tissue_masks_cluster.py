# So nnunetv2 shuts up with warning messages that are not relevant
import os
# /data/scratch/r111888
os.environ["nnUNet_raw"] = "C:/tmp/nnunet_raw"
os.environ["nnUNet_preprocessed"] = "C:/tmp/nnunet_preprocessed"
os.environ["nnUNet_results"] = "C:/tmp/nnunet_results"

# You have to cd into the tissue masking folder and do a pip install -e .
# Add to folder containing tissue mask nnunet and models folder a init.py with from nnunetv2 import *

from src.tissue_masking.nnunetv2.inference.predict_tissue import TissueNNUnetPredictor

from PIL import Image
from tiatoolbox.wsicore import WSIReader
from openslide import OpenSlide
from pathlib import Path
import os
from tqdm import tqdm
import torch
from loguru import logger
import pandas as pd

def get_all_eligible_wsi_ids() -> list:
    df = pd.read_csv('./src/config/dhgpscores_allslides.csv', sep=';', encoding='latin-1')
    df = df[['Cohort','EMC_flnm','Resection','Tissue','dHGP', 'Preop_CTx']]
    # Filter out Chemoexposed patients
    df = df[df['Cohort']=='EMC']
    df = df[(df['Resection'] =='PA_nummer_CRLM') | (df['Resection'] =='Stage2_PA_nummer_CRLM') ]
    df = df[df['Preop_CTx'] == 'No'] # No Chemotherapy
    df = df[df['Tissue']=='CRLM']
    mask = df['dHGP'] != '-'
    df = df[mask]
    df['dHGP'] = df['dHGP'].astype(pd.Int16Dtype())
    # Define the three sampling groups
    # dhgp_0 = df[df['dHGP'] == 0]
    # dhgp_1_99 = df[(df['dHGP'] >= 1) & (df['dHGP'] <= 99)]
    # dhgp_100 = df[df['dHGP'] == 100]
    print(len(df))
    return df["EMC_flnm"].tolist()

def create_downsampled_thumbnails(
    inputdir: str | Path = "./input", outputdir: str | Path = "./output"
):
    """
    Takes in a path of whole slide images, and then create downsampled thumbnails in 10mpp resolution to later pass them through the tissue masking
    """
    if not os.path.exists(str(inputdir)):
        raise ModuleNotFoundError("Given input folder does not exist")
    if not os.path.exists(str(outputdir)):
        raise ModuleNotFoundError("Given output folder does not exist")

    if type(inputdir) != Path:
        inputdir = Path(inputdir)
    if type(outputdir) != Path:
        outputdir = Path(outputdir)

    wsi_fps = list(inputdir.glob("*.mrxs"))
    matches_fps = [
        wsi_fp
        for wsi_fp in wsi_fps
        # Check if a directory with the same stem exists in the file's parent directory
        if wsi_fp.parent.joinpath(wsi_fp.stem).is_dir()
    ]
    for fp in matches_fps:
        if "9736" in str(fp):
            print('its here')
    if len(matches_fps) < 1:
        raise RuntimeError("No MRXS Files found in given input folder")
    print(
        f"Found a total of {len(matches_fps)} Whole Slide Images. Beginning Downsampling"
    )

    for wsi_path in tqdm(matches_fps, desc="Whole Slide Images Processed", ncols=100):
        out_img_path = str(outputdir.joinpath(f"{wsi_path.stem}.png"))
        if os.path.exists(out_img_path):
            logger.info(f"Skipping {wsi_path.stem} since already processed")
            continue
        wsi = OpenSlide(filename=str(wsi_path))
        start_x = int(wsi.properties["openslide.bounds-x"])
        start_y = int(wsi.properties["openslide.bounds-y"])
        # Get the bounds you want to crop
        width = int(wsi.properties["openslide.bounds-width"])
        height = int(wsi.properties["openslide.bounds-height"])
        tia_wsi = WSIReader.open(str(wsi_path))
        ds_array = tia_wsi.read_bounds(
            bounds=(start_x, start_y, start_x + width, start_y + height),
            resolution=10,
            units="mpp",
        )
        image = Image.fromarray(ds_array)
        image.save(out_img_path)

def create_downsampled_thumbnails_fromlist(
     id_list:list[str], inputdir: str | Path = "./input",outputdir: str | Path = "./output"
):
    """
    Takes in a path of whole slide images, and then create downsampled thumbnails in 10mpp resolution to later pass them through the tissue masking
    """
    if not os.path.exists(str(inputdir)):
        raise ModuleNotFoundError("Given input folder does not exist")
    if not os.path.exists(str(outputdir)):
        raise ModuleNotFoundError("Given output folder does not exist")

    if type(inputdir) != Path:
        inputdir = Path(inputdir)
    if type(outputdir) != Path:
        outputdir = Path(outputdir)

    # wsi_fps = list(inputdir.glob("*.mrxs"))
    wsi_fps = [Path(inputdir+"/"+id+".mrxs")]
    matches_fps = [
        wsi_fp
        for wsi_fp in wsi_fps
        # Check if a directory with the same stem exists in the file's parent directory
        if wsi_fp.parent.joinpath(wsi_fp.stem).is_dir()
    ]
    if len(matches_fps) < 1:
        raise RuntimeError("No MRXS Files found in given input folder")
    print(
        f"Found a total of {len(matches_fps)} Whole Slide Images. Beginning Downsampling"
    )

    for wsi_path in tqdm(matches_fps, desc="Whole Slide Images Processed", ncols=100):
        out_img_path = str(outputdir.joinpath(f"{wsi_path.stem}.png"))
        if os.path.exists(out_img_path):
            logger.info(f"Skipping {wsi_path.stem} since already processed")
            continue
        wsi = OpenSlide(filename=str(wsi_path))
        start_x = int(wsi.properties["openslide.bounds-x"])
        start_y = int(wsi.properties["openslide.bounds-y"])
        # Get the bounds you want to crop
        width = int(wsi.properties["openslide.bounds-width"])
        height = int(wsi.properties["openslide.bounds-height"])
        tia_wsi = WSIReader.open(str(wsi_path))
        ds_array = tia_wsi.read_bounds(
            bounds=(start_x, start_y, start_x + width, start_y + height),
            resolution=10,
            units="mpp",
        )
        image = Image.fromarray(ds_array)
        image.save(out_img_path)


def create_tissue_masks(inputdir: str | Path, outputdir: str | Path):
    if not os.path.exists(str(inputdir)):
        raise FileNotFoundError("Given input folder does not exist")
    if not os.path.exists(str(outputdir)):
        raise FileNotFoundError("Given output folder does not exist")

    if type(inputdir) != Path:
        inputdir = Path(inputdir)
    if type(outputdir) != Path:
        outputdir = Path(outputdir)

    # Check for PNGs
    ds_image_paths = list(inputdir.glob("*.png"))

    if len(ds_image_paths) < 1:
        raise RuntimeError("No PNG downsampled files found in given input folder")

    # TODO Prefilter the ones that were already created

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Init predictor
    predictor = TissueNNUnetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,  # or "cpu" if no GPU
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    # Load the 10 µm model
    predictor.initialize_from_trained_tissue_model_folder(
        "./src/tissue_masking/models/trained_on_10um",
        use_folds="all",
        checkpoint_name="checkpoint_10um.pth",
    )

    # Lite postprocessing config
    pp_cfg = dict(fill_holes=True, min_area_rel=0.002)

    # Run segmentation
    predictor.predict_tissue_from_files(
        str(inputdir),
        str(outputdir),
        save_probabilities=False,
        overwrite=False,
        suffix=None,
        extension=None,
        exclude=None,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
        binary_01=False,  # <- outputs [0,255] instead of [0,1]
        keep_parent=False,
        lowres=False,  # <- stick to 10 µm
        pp_cfg=pp_cfg,
    )


if __name__ == "__main__":
    # python -m src.preproc_scripts.create_tissue_masks
    # ds_input = "D:/tfm_data/unlabeled_wsi_new"
    ds_output = "./10mpp_unlabeled_all"
    tissue_output = "./outputs/binary_masks_unlabeled"
    # create_downsampled_thumbnails(inputdir=ds_input, outputdir=ds_output)
    create_tissue_masks(inputdir=ds_output, outputdir=tissue_output)
