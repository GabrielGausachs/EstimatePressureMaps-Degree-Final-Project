import torch
import torch.nn as nn
import math

from Utils.config import (
    EVALUATION,
    DEVICE,
)

# Pixel Wise Resampling Loss

class PWRSWtL(nn.Module):
    def __init__(self, lambda_L2):
        super(PWRSWtL, self).__init__()
        self.lambda_L2 = lambda_L2

    def forward(self, src, tar):
        batch_size = tar.size(0)

        # Pixel density for all the batch
        # Defineixo els edges de l'histograma entre 0 i el valor maxim del batch d'1 en 1
        # El primer edge serà de 0-1, el segon de 0-2,...
        bin_edges = torch.arange(0, math.ceil(
            tar.max())+1, 1, dtype=torch.float32)

        tar_cpu = tar.view(-1).cpu()
        # Fem l'histograma
        hist = torch.histogram(tar_cpu, bins=bin_edges)

        # Funció de densitat
        f_dems = hist[0]/tar.numel()

        # Reverse pixel density i normalitzem
        weights = self.lambda_L2 / (f_dems+1e-2)
        weights = weights / weights.sum()

        # Per cada interval de valors, tenim el seu weight en la loss.
        # Valors no comuns, weights més grans.
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
        #loss = (self.lambda_L2 * mask_tensor * diff_sq).sum()

        return loss


# Higher Values Loss
class HVLoss(nn.Module):
    def __init__(self, lambda_L2):
        super(HVLoss, self).__init__()
        self.lambda_L2 = lambda_L2

    def forward(self, src, tar):

        # Higher values, higher weights

        min_value = 0.1

        # Compute the range of pixel values
        value_range = self.lambda_L2 - min_value

        # Compute weights for each pixel value
        weights = (tar - tar.min()) / (tar.max() -
                                       tar.min()) * value_range + min_value

        # Create a mask tensor with the computed weights
        mask_tensor = weights.clone()

        diff_sq = (src - tar) ** 2

        loss = (mask_tensor * diff_sq).mean()

        return loss
