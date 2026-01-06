import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def multiclass_dice_loss(pred, target, smooth=1):
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    dice_per_class = []

    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c]
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        
        # Mask out images where class c does not appear
        mask = target_c.sum(dim=(1, 2)) > 0
        if mask.sum() == 0:
            continue  # skip if class not present in any image
        
        dice_c = (2. * intersection[mask] + smooth) / (union[mask] + smooth)
        dice_per_class.append(dice_c.mean())

    if len(dice_per_class) == 0:
        return torch.tensor(0.0, device=pred.device)
    
    return 1 - torch.stack(dice_per_class).mean()

def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(
        labels.size()[0], classes, labels.size()[2], labels.size()[3], labels.size()[4]
    ).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (output_flat.sum() + target_flat.sum() + self.smooth)
        )
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self):
        super(CE_DiceLoss, self).__init__()
        self.dice = DiceLoss(smooth=1e-5)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class EntropyLoss(nn.Module):
    """
    Entropy loss for probabilistic prediction vectors
    input: batch_size x channels x h x w x d
    output: batch_size x 1 x h x w x d
    """

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, v):
        assert v.dim() == 5
        n, c, h, w, d = v.size()
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (
            n * h * w * d * np.log2(c)
        )

import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalTverskyLoss(nn.Module):
    """
    CORRECTED Multi-Class Asymmetric Focal Tversky Loss.
    Uses (1-mTI)^(1-gamma) for enhancement, as per Eq. 21.
    """
    def __init__(self, delta=0.6, gamma=0.5, background_class_idx=0, epsilon=1e-7):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.background_class_idx = background_class_idx
        self.epsilon = epsilon

    def forward(self, y_pred_prob, y_true_onehot):
        if y_pred_prob.dim() == 4: # 2D (B, C, H, W)
            spatial_axis = (2, 3)
        elif y_pred_prob.dim() == 5: # 3D (B, C, D, H, W)
            spatial_axis = (2, 3, 4)
        else:
            raise ValueError("Input tensor must be 4D or 5D")

        tp = torch.sum(y_true_onehot * y_pred_prob, dim=spatial_axis)
        fn = torch.sum(y_true_onehot * (1 - y_pred_prob), dim=spatial_axis)
        fp = torch.sum((1 - y_true_onehot) * y_pred_prob, dim=spatial_axis)
        
        mTI = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
        
        tversky_loss_base = 1 - mTI
        
        # --- CORRECTED EXPONENT ---
        # Use (1 - gamma) for enhancement, as per Eq. 21
        focal_enhancement = torch.pow(tversky_loss_base, 1.0 - self.gamma)
        
        background_mask = torch.zeros_like(tversky_loss_base, dtype=torch.bool)
        background_mask[:, self.background_class_idx] = True
        
        final_loss_per_class = torch.where(
            background_mask,
            tversky_loss_base,
            focal_enhancement
        )
        
        return final_loss_per_class.mean()

class AsymmetricFocalLoss(nn.Module):
    """
    CORRECTED Multi-Class Asymmetric Focal (Cross-Entropy) Loss.
    Uses (1-p_t)^gamma for suppression, as per Eq. 20.
    """
    def __init__(self, delta=0.6, gamma=0.5, rare_class_idx=3, epsilon=1e-7):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.rare_class_idx = rare_class_idx
        self.epsilon = epsilon

    def forward(self, y_pred_prob, y_pred_log_prob, y_true_onehot):
        num_classes = y_pred_prob.shape[1]
        
        cross_entropy = -y_true_onehot * y_pred_log_prob
        
        # --- CORRECTED EXPONENT ---
        # Use (gamma) for suppression, as per Eq. 20
        focal_suppression = torch.pow(1 - y_pred_prob, self.gamma)
        
        focal_suppression[:, self.rare_class_idx, ...] = 1.0
        
        focal_loss_per_pixel = focal_suppression * cross_entropy
        
        delta_weights = torch.full((num_classes,), (1 - self.delta) / (num_classes - 1), 
                                   device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        delta_weights[self.rare_class_idx] = self.delta
        
        delta_weights = delta_weights.view(1, num_classes, *([1] * (y_pred_prob.dim() - 2)))
        
        final_focal_loss = delta_weights * focal_loss_per_pixel
        
        return final_focal_loss.mean()

class AsymmetricUnifiedFocalLoss(nn.Module):
    """
    PyTorch multi-class implementation of the Asymmetric Unified Focal Loss
    with CORRECTED exponents.
    
    This loss expects *LOGITS* as input.
    """
    def __init__(self, lambda_weight=0.5, delta=0.6, gamma=0.5, 
                 num_classes=4, rare_class_idx=3, background_class_idx=0, epsilon=1e-7):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.num_classes = num_classes
        
        self.asym_focal_loss = AsymmetricFocalLoss(
            delta=delta, 
            gamma=gamma, 
            rare_class_idx=rare_class_idx, 
            epsilon=epsilon
        )
        
        self.asym_tversky_loss = AsymmetricFocalTverskyLoss(
            delta=delta, 
            gamma=gamma, 
            background_class_idx=background_class_idx, 
            epsilon=epsilon
        )

    def forward(self, logits, y_true):
        y_pred_prob = F.softmax(logits, dim=1)
        y_pred_log_prob = F.log_softmax(logits, dim=1)
        
        y_true_onehot = F.one_hot(y_true, num_classes=self.num_classes).permute(0, -1, *range(1, y_true.dim()))
        y_true_onehot = y_true_onehot.float()

        loss_focal = self.asym_focal_loss(y_pred_prob, y_pred_log_prob, y_true_onehot)
        loss_tversky = self.asym_tversky_loss(y_pred_prob, y_true_onehot)
        
        total_loss = (self.lambda_weight * loss_tversky) + ((1 - self.lambda_weight) * loss_focal)
        
        # Return all three for logging
        return total_loss, loss_tversky, loss_focal