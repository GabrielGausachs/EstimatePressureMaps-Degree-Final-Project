import torch
import torch.nn as nn

class PWRSWtL(nn.Module):
    def __init__(self, lambda_L2):
        super(PWRSWtL, self).__init__()
        self.lambda_L2 = lambda_L2

    def forward(self, src, tar):
        # Calculate the density function of the pixel value y
        p_y = torch.histc(tar, bins=256, min=0, max=255) / tar.numel()

        # Calculate the weight function
        weight = self.lambda_L2 / (p_y + 1e-12)  # Adding a small epsilon to avoid division by zero

        # Compute the weighted L2 loss
        loss = (weight * (src - tar) ** 2).mean()

        return loss
