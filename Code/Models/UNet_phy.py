import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# A U-Net with 4 blocks of 2 convolutional layers each block and with skip connections.
# With A physical vector added that does the encoder separately, then concats with
# The output of the encoder of the the convolutional blocks and does the decoder jointly


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET_phy(nn.Module):
    def __init__(
            self, in_channels=3, in_channels_phy=11, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET_phy, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET array
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Down part of UNET physical vector
        phy_fc = nn.ModuleList()
        phy_fc.append(nn.Linear(in_channels_phy, 64))
        phy_fc.append(nn.ReLU(True))
        phy_fc.append(nn.Dropout(0.5))
        phy_fc.append(nn.Linear(64, 64))
        phy_fc.append(nn.ReLU(True))
        phy_fc.append(nn.Dropout(0.5))
        phy_fc.append(nn.Linear(64, 64)
                      )     # quants features de sortida?
        self.phyNet = nn.Sequential(*phy_fc)

        # Up part of UNET
        first_iteration = False
        for feature in reversed(features):
            if first_iteration:
                self.ups.append(
                    nn.ConvTranspose2d(
                        1035, feature, kernel_size=2, stride=2,
                    )
                )
                first_iteration = False
            else:
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                    )
                )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x, x_phy):
        skip_connections = []

        # Down part of the Array
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Down part of the Physical vector
        x_phy = x_phy.to(torch.float32)
        x_phy = self.phyNet(x_phy)
        x_phy = x_phy.unsqueeze(-1).unsqueeze(-1)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = torch.cat((x, x_phy.expand(-1, -1, 192, 84)), dim=1)

        return self.final_conv(x)


# afegir això: x = torch.cat((x, x_phy.expand(-1, -1, 12, 5)), dim=1)
# Per tant tindrem x.shape = (32,1035,12,5)
# Hauriem de canviar les primeres convolucions transposades (self.ups[idx](x))
# Perque rebin 1035 canals i treguin 512.
# Sinó, podem anar fent 1035 - 518 - 259 - 130 - 65 - 1
