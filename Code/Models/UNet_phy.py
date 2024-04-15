import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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
        phy_fc.append(nn.Linear(in_channels_phy, 10))
        phy_fc.append(nn.ReLU(True))
        phy_fc.append(nn.Dropout(0.5))
        phy_fc.append(nn.Linear(10, 10))
        phy_fc.append(nn.ReLU(True))
        phy_fc.append(nn.Dropout(0.5))
        phy_fc.append(nn.Linear(10, 1))     # quants features de sortida?
        self.phyNet = nn.Sequential(*phy_fc)


        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x,x_phy):
        skip_connections = []

        # Down part of the Array
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Down part of the Physical vector
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

        return self.final_conv(x)