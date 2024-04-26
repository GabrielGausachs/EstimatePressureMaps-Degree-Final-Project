import torch
import torch.nn as nn


class PerCS(nn.Module):
    def __init__(self):
        super(PerCS, self).__init__()

    def forward(self, src, tar):

        diff = torch.abs(src-tar)

        max_value = torch.max(tar)
        print(max_value)

        count = torch.sum(diff < (max_value*0.01)).item()

        pcs = count / tar.numel()

        return pcs
