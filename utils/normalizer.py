import torch
import tiatoolbox
import openslide
from tiatoolbox.wsicore import WSIReader
from PIL import Image
import numpy as np
# from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer
# from torchstain.torch.normalizers.reinhard import TorchReinhardNormalizer
from torchstain.torch.utils import cov, percentile
import os
from tiatoolbox.tools.stainnorm import get_normalizer as get_tia_normalizer, load_stain_matrix

# Helper function - no changes
def numpy_to_torch(img: np.ndarray) -> torch.Tensor:
    """Convert a numpy array [H, W, C] to a torch tensor [C, H, W]."""
    return torch.from_numpy(img).permute(2, 0, 1)


class TorchRobustMacenkoNormalizer():
    """
    Inherits from torchstain's normalizer and fixes the GPU crash by
    overriding the method that calls the unstable code. Also includes
    robust fallback logic for pale patches. Adds save/load functionality.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.HERef = torch.tensor(
            [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], device=self.device
        )
        self.maxCRef = torch.tensor([1.9705, 1.0308], device=self.device)
        self.updated_lstsq = hasattr(torch.linalg, "lstsq")
        self.exception_counter = 0
        self.success_counter = 0

    def __convert_rgb2od(self, I, Io, beta):
        """
        (This is a helper for __compute_matrices)
        Permutes to [H, W, C] and converts to Optical Density.
        """
        I = I.permute(1, 2, 0)  # Change from [C, H, W] to [H, W, C]
        I = I.reshape(-1, 3)

        OD = -torch.log((I.float() + 1) / Io)
        ODhat = OD[~torch.any(OD < beta, dim=1)]
        return OD, ODhat
    
    def __find_HE(self, ODhat, eigvecs, alpha):
        """
        (This is a helper for __compute_matrices)
        Finds the H&E stain vectors.
        """
        t = torch.matmul(ODhat, eigvecs)
        phi = torch.atan2(t[:, 1], t[:, 0])

        # minPhi = percentile(phi, alpha)
        # maxPhi = percentile(phi, 100 - alpha)
        # q99 = torch.tensor(0.99, device=self.device)
        minPhi = torch.quantile(phi,alpha/100)
        maxPhi = torch.quantile(phi, 1-alpha/100)

        vMin = torch.matmul(
            eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))
        ).unsqueeze(1)
        vMax = torch.matmul(
            eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))
        ).unsqueeze(1)

        HE = torch.where(
            vMin[0] > vMax[0],
            torch.cat((vMin, vMax), dim=1),
            torch.cat((vMax, vMin), dim=1),
        )
        return HE
    
    def __find_concentration(self, OD, HE):
        """
        (This is a helper for __compute_matrices)
        Finds the stain concentrations.
        """
        # This is the new implementation from the commit
        return torch.linalg.pinv(HE) @ OD.T

    def __compute_matrices(self, I, Io, alpha, beta):
        """
        (This is a helper for normalize)
        Computes the H&E matrix and concentrations.
        """
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)
        
        # This check is from your robust class, you should keep it
        if ODhat.shape[0] < 2:
            raise torch.linalg.LinAlgError(
                "Not enough tissue found to compute stain matrix."
            )
            
        _, eigvecs = torch.linalg.eigh(cov(ODhat.T))
        eigvecs = eigvecs[:, [1, 2]]

        HE = self.__find_HE(ODhat, eigvecs, alpha)
        C = self.__find_concentration(OD, HE)

        q99 = torch.tensor(0.99, device=self.device)

        maxC = torch.stack([
            torch.quantile(C[0, :], q99), 
            torch.quantile(C[1, :], q99)
        ])
        # maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        """
        Fits the normalizer and stores HERef and maxCRef.
        """
        I = I.to(self.device)
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)
        self.HERef = HE
        self.maxCRef = maxC

    def save_fit(self, filepath):
        """
        Saves the fitted HERef and maxCRef to a file.
        """
        torch.save({"HERef": self.HERef, "maxCRef": self.maxCRef}, filepath)

    def load_fit(self, filepath):
        """
        Loads pre-fitted HERef and maxCRef from a file.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.HERef = checkpoint["HERef"].to(self.device)
        self.maxCRef = checkpoint["maxCRef"].to(self.device)


    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=False):
        """
        Normalizes an image with GPU-stable logic and robust fallback.
        """
        if not torch.is_tensor(I):
            I_tensor = numpy_to_torch(I).to(self.device)
        else:
            I_tensor = I.to(self.device)
        c, h, w = I_tensor.shape

        try:
            # 1. Try to compute stain matrix from the current patch
            # This calls your overridden __compute_matrices
            HE, C, maxC = self.__compute_matrices(I_tensor, Io, alpha, beta)

            # 2. If it succeeds, normalize using the computed matrix (HE)
            #    and the pre-fitted target (self.HERef).
            
            # Normalize stain concentrations
            C = C * (self.maxCRef.unsqueeze(-1) / maxC.unsqueeze(-1))

            # Recreate normalized image using *pre-fitted* HERef
            Inorm = torch.mul(Io, torch.exp(torch.matmul(-self.HERef, C)))
            # Inorm[Inorm > 255] = 255
            Inorm = torch.clamp(Inorm, 0, 255)
            # Reshape and permute to [C, H, W]
            Inorm = Inorm.T.reshape(h, w, c).int().permute(2, 0, 1)

            H, E = None, None
            if stains:
                H = torch.mul(
                    Io,
                    torch.exp(
                        torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))
                    ),
                )
                H[H > 255] = 255
                H = H.T.reshape(h, w, c).int() # [H, W, C] tensor

                E = torch.mul(
                    Io,
                    torch.exp(
                        torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))
                    ),
                )
                E[E > 255] = 255
                E = E.T.reshape(h, w, c).int() # [H, W, C] tensor
            self.success_counter+=1
            return Inorm, H, E

        except torch.linalg.LinAlgError:
            # 3. If it fails (e.g., all-white patch), use the pre-fitted matrices
            # This is the robust fallback logic
            
            OD = -torch.log((I_tensor.permute(1, 2, 0).float() + 1) / Io)
            OD_reshaped = OD.reshape(-1, 3).T
            
            # Get concentrations using the *pre-fitted* matrix
            source_concentrations = torch.linalg.pinv(self.HERef) @ OD_reshaped
            
            # Check for zero concentrations
            max_c_source, _ = torch.quantile(source_concentrations, 0.99, dim=1)
            if torch.any(max_c_source <= 1e-3): 
                self.exception_counter += 1
                return I_tensor.int(), None, None

            # Scale concentrations to the *pre-fitted* max
            source_concentrations *= (self.maxCRef / max_c_source).unsqueeze(-1)
            
            # Reconstruct the image
            trans = Io * torch.exp(-1 * torch.matmul(self.HERef, source_concentrations))
            trans = torch.clamp(trans, 0, 255)
            trans = trans.T.reshape(h, w, c)
            # trans[trans > 255] = 255
            # trans[trans < 0] = 0
            
            Inorm = trans.permute(2, 0, 1).to(I_tensor.dtype) # [C, H, W] tensor

            H, E = None, None
            if stains:
                H = torch.mul(
                    Io,
                    torch.exp(
                        torch.matmul(-self.HERef[:, 0].unsqueeze(-1), source_concentrations[0, :].unsqueeze(0))
                    ),
                )
                H[H > 255] = 255
                H = H.T.reshape(h, w, c).int() # [H, W, C] tensor

                E = torch.mul(
                    Io,
                    torch.exp(
                        torch.matmul(-self.HERef[:, 1].unsqueeze(-1), source_concentrations[1, :].unsqueeze(0))
                    ),
                )
                E[E > 255] = 255
                E = E.T.reshape(h, w, c).int() # [H, W, C] tensor
            self.exception_counter+=1
            return Inorm, H, E


def get_wsi_normalizer(
    target_size: int = 512,
    # target_level: int = 2,
    # target_fp: str = "D:/tfm_data/labeled_wsi/H15-16121-IA.mrxs",
    target_level: int = 1,
    target_fp: str = "D:/tfm_data/labeled_wsi/H18-24578-IIB.mrxs",
    device: str = "cuda",
    get_prefitted: bool = False,
    # fit_filepath: str = "./src/pretrained_models/normalizer_fit_1811_lvl3.pt",
    fit_filepath:str = "./src/pretrained_models/normalizer_fit_1911.pt",
    save = False
):
    """
    Fits or loads a pre-fitted robust, GPU-stable normalizer on a reference WSI.
    If get_prefitted=True and fit_filepath exists, loads the saved parameters.
    Otherwise, fits the normalizer and saves the parameters if fit_filepath is provided.
    """
    normalizer = TorchRobustMacenkoNormalizer(device=device)

    if get_prefitted and os.path.exists(fit_filepath):
        normalizer.load_fit(fit_filepath)
        return normalizer

    target_wsi_os = openslide.OpenSlide(filename=target_fp)
    start_x = int(target_wsi_os.properties.get("openslide.bounds-x", 0))
    start_y = int(target_wsi_os.properties.get("openslide.bounds-y", 0))
    tia_wsi = WSIReader.open(target_fp)
    ds_levels = target_wsi_os.level_downsamples
    ds_ratio = int(ds_levels[target_level])
    mpp_x = float(target_wsi_os.properties["openslide.mpp-x"])
    mpp_y = float(target_wsi_os.properties["openslide.mpp-y"])
    # roi_x_rel, roi_y_rel = 8340, 11340
    roi_x_rel, roi_y_rel = 7440, 15440
    scaled_roi_x, scaled_roi_y = int(roi_x_rel / mpp_x), int(roi_y_rel / mpp_y)
    start_roi_x, start_roi_y = start_x + scaled_roi_x, start_y + scaled_roi_y
    roi_np = tia_wsi.read_bounds(
        bounds=(
            start_roi_x,
            start_roi_y,
            start_roi_x + target_size * ds_ratio,
            start_roi_y + target_size * ds_ratio,
        ),
        resolution=target_level,
        units="level",
    )
    roi_torch = numpy_to_torch(roi_np)

    print("Fitting normalizer...")
    normalizer.fit(roi_torch)
    print("Normalizer fitted successfully.")

    if fit_filepath and save:
        print(f"Saving fitted normalizer to {fit_filepath}...")
        normalizer.save_fit(fit_filepath)
        print("Fitted normalizer saved successfully.")

    return normalizer