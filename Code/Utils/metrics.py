import torch
import torch.nn as nn
from pytorch_msssim import ssim


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


class SSIM(nn.Module):
    def __init__(self, max_val=1.0):
        super(SSIM, self).__init__()
        self.max_val = max_val

    def forward(self, y_pred, y_true):
        return ssim(y_true, y_pred, data_range=self.max_val, size_average=True)

class MSEeff(nn.Module):
    def __init__(self):
        super(MSEeff,self).__init__()

    def forward(self,src,tar):
        mask = tar > 0.05
        loss = ((src[mask]-tar[mask])**2).mean()
        return loss
