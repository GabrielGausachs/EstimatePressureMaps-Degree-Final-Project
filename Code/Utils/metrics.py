import torch
import torch.nn as nn


class PerCS(nn.Module):
    def __init__(self):
        super(PerCS, self).__init__()

    def forward(self, src, tar):

        diff = torch.abs(src-tar)

        reshaped_tar = tar.view(tar.size(0), -1)

        max_values, _ = torch.max(reshaped_tar, dim=1)

        thres = max_values.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
            tar.size(0), tar.size(1), tar.size(2), tar.size(3))*0.025

        count = torch.sum(diff < thres).item()

        pcs = count / tar.numel()

        return pcs
