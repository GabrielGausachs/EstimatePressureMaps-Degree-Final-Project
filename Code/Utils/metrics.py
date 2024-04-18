import torch
import torch.nn as nn
import math
import numpy as np

class PerCS(nn.Module):
    def __init__(self):
        super(PerCS, self).__init__()
    
    def forward(self,src,tar):
        # Fer la resta entre src i tar
        # Fer la mean difference d'aquells valors que la difer
        pass

