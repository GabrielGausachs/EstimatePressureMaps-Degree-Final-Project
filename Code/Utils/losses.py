import torch
import torch.nn as nn
import math
import numpy as np

from Utils.config import (
    EVALUATION,
    DEVICE,
)

from pytorch_msssim import ssim

class SSIMLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(SSIMLoss, self).__init__()
        self.max_val = max_val

    def forward(self, y_pred, y_true):
        return 1.0 - ssim(y_true, y_pred, data_range=self.max_val, size_average=True)


class UVLoss(nn.Module):
    def __init__(self, lambda_L2):
        super(UVLoss, self).__init__()
        self.lambda_L2 = lambda_L2

    def forward(self, src, tar):
        batch_size = tar.size(0)

        # Pixel density for all the batch
        # I define the edges of the histogram between 0 and the max value of the batch
        bin_edges = torch.arange(0, math.ceil(
            tar.max())+1, 1, dtype=torch.float32)

        tar_cpu = tar.view(-1).cpu()

        # Do the histogram
        hist = torch.histogram(tar_cpu, bins=bin_edges)

        # Desnity function
        f_dems = hist[0]/tar.numel()

        # Reverse pixel density and normalize
        weights = self.lambda_L2 / (f_dems+1e-2)
        weights = weights / weights.sum()

        # For every interval of values, we have the weight in the loss function
        # No common values, higher weight
        pixel_weights = {}
        for pixel_value, weight_value in enumerate(weights):
            pixel_weights[pixel_value] = weight_value.item()

        tensor_size = (batch_size, 1, 192, 84)

        if EVALUATION:
            tensor_size = (1, 192, 84)

        # Create mask tensor with zeros
        mask_tensor = torch.zeros(tensor_size, device=DEVICE)

        # Fill mask tensor with pixel-wise probabilities
        for pixel_value, probability in pixel_weights.items():
            mask_tensor[(pixel_value <= tar) & (
                tar < pixel_value+1)] = probability

        # Get the diff between output and target
        diff_sq = (src - tar) ** 2

        # Get the loss with mean
        loss = (mask_tensor * diff_sq).mean()

        # Get the loss with sum
        # loss = (self.lambda_L2 * mask_tensor * diff_sq).sum()

        return loss


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