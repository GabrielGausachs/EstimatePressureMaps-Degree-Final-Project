import torch
import torch.nn as nn

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
        x = torch.cat([x, x5], dim=1)
        x = self.up_conv1(x)
        x = self.up_transpose2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        x = self.up_transpose3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv3(x)
        
        # Output
        x = self.out_conv(x)
        return x


# Example usage:
# Define the model
model = UNet(in_channels=1, out_channels=1)  # Assuming grayscale images (1 channel) and predicting 1 channel output
# Define loss function
criterion = nn.MSELoss()
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass
input_image = torch.randn(1, 1, 256, 256)  # Example input image shape: (batch_size, channels, height, width)
output_map = model(input_image)

# Calculate loss
target_map = torch.randn(1, 1, 256, 256)  # Example target output shape: (batch_size, channels, height, width)
loss = criterion(output_map, target_map)

# Backward pass and optimization step
optimizer.zero_grad()
loss.backward()
optimizer.step()
