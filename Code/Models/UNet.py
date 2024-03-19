import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down_conv1 = DoubleConv(in_channels, 64)
        self.down_conv2 = DoubleConv(64, 128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_transpose1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up_transpose2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up_transpose3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.down_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.down_conv4(x6)
        
        # Decoder
        x = self.up_transpose1(x7)
        x5 = self.upsample_and_concat(x5, x)
        x = self.up_conv1(x5)
        x = self.up_transpose2(x)
        x = self.upsample_and_concat(x3, x)
        x = self.up_conv2(x)
        x = self.up_transpose3(x)
        x = self.upsample_and_concat(x1, x)
        x = self.up_conv3(x)
        
        # Output
        x = self.out_conv(x)
        return x

    def upsample_and_concat(self, x1, x2):
        # Upsample x1 to match the dimensions of x2
        _, _, H, W = x2.size()
        x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=True)
        # Concatenate along the channel dimension
        return torch.cat([x1, x2], dim=1)
