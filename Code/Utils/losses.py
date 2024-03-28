import torch
import torch.nn as nn

from Utils.config import (
    EVALUATION,
    DEVICE,
)

class PWRSWtL(nn.Module):
    def __init__(self, lambda_L2):
        super(PWRSWtL, self).__init__()
        self.lambda_L2 = lambda_L2

    def forward(self, src, tar):
        batch_size = tar.size(0)

        # Pixel density for all the batch
        p_y = torch.histc(tar.view(-1), bins=256, min=0, max=255) / (tar.numel() * batch_size)

        # Reverse pixel density and normalize
        weight = 1 / (p_y + 1e-12)
        weight = weight / weight.sum()

        pixel_probabilities = {}
        for pixel_value, probability in enumerate(weight):
            pixel_probabilities[pixel_value] = probability.item()


        tensor_size = (128,1,192,84)
        
        if EVALUATION:
            tensor_size = (1,192,84)

        # Create mask tensor with zeros
        mask_tensor = torch.zeros(tensor_size,device = DEVICE)

        # Fill mask tensor with pixel-wise probabilities
        for pixel_value, probability in pixel_probabilities.items():
            mask_tensor[tar == pixel_value] = probability

        # Get the diff between output and target
        diff_sq = (src - tar) ** 2

        # Get the loss with mean
        #loss = (self.lambda_L2 * mask_tensor * diff_sq).mean()

        # Get the loss with sum
        loss = (self.lambda_L2 * mask_tensor * diff_sq).sum()

        return loss
