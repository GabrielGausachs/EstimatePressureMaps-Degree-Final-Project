import torch
import torch.nn as nn

class PWRSWtL(nn.Module):
    def __init__(self, lambda_L2):
        super(PWRSWtL, self).__init__()
        self.lambda_L2 = lambda_L2

    def forward(self, src, tar):
        batch_size = tar.size(0)  # Obtener el tamaño del lote

        # Calcular la densidad de píxeles para todo el lote tar
        p_y = torch.histc(tar.view(-1), bins=256, min=0, max=255) / (tar.numel() * batch_size)

        # Invertir la densidad de píxeles y normalizarla
        weight = 1 / (p_y + 1e-12)
        weight = weight / weight.sum()

        # Expandir las dimensiones del tensor de pesos para que coincida con las dimensiones de src y tar
        weight = weight.view(1, 1, 1, 256).to(tar.device)

        # Calcular la diferencia al cuadrado entre src y tar
        diff_sq = (src - tar) ** 2

        # Calcular la pérdida ponderada para todo el lote
        loss = (self.lambda_L2 * weight * diff_sq).mean()

        return loss
