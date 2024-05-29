import torch
import torch.nn as nn
import math
import numpy as np
from pytorch_msssim import ssim

from Utils.config import (
    DEVICE,
)

class SSIMLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(SSIMLoss, self).__init__()
        self.max_val = max_val

    def forward(self, y_pred, y_true):
        return 1.0 - ssim(y_true, y_pred, data_range=self.max_val, size_average=True)


# Higher Values Loss
class HVLoss(nn.Module):
    def __init__(self):
        super(HVLoss, self).__init__()

    def forward(self, src, tar):

        # Higher values, higher weights

        min_value = 1
        max_value = 10

        # Compute the range of pixel values
        value_range = max_value - min_value

        # Compute weights for each pixel value
        weights = (tar - tar.min()) / (tar.max() -
                                       tar.min()) * value_range + min_value

        # Create a mask tensor with the computed weights
        mask_tensor = weights.clone()

        diff_sq = (src - tar) ** 2

        loss = (mask_tensor * diff_sq).mean()

        return loss


class PhyLoss(nn.Module):
    def __init__(self):
        super(PhyLoss, self).__init__()

    def forward(self, src, tar):

        area_m = 1.03226 / 10000
        desire_weight = area_m * (torch.sum(tar, dim=(1, 2))*1000) / 9.81
        output_weight = area_m * (torch.sum(src, dim=(1, 2))*1000) / 9.81

        loss = (desire_weight - output_weight)**2

        loss = loss.mean()

        return loss