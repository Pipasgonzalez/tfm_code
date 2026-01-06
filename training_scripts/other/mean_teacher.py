from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from monai.networks.nets.swin_unetr import SwinUNETR
from typing import Literal


def get_teacher(type: Literal["unet", "unet++", "swinunet"], nclasses):
    if type == "unet":
        return BasicUNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=nclasses,
        )
    elif type == "unet++":
        return BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=3,
            out_channels=nclasses,
        )
    elif type == "swinunet":
        return SwinUNETR(
            in_channels=3,  # RGB input
            out_channels=nclasses,  # 7 segmentation classes
            patch_size=2,  # changed this from 4 to 2
            feature_size=48,  # moderate channel width
            depths=(2, 2, 2, 2),  # number of layers per stage
            num_heads=(3, 6, 12, 24),  # multi-head attention per stage
            window_size=7,  # attention window
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            norm_name="instance",
            patch_norm=False,
            use_checkpoint=True,  # save GPU memory
            spatial_dims=2,  # 2D images
        )
