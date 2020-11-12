import torch
import torch.nn as nn


def double_conv(in_channels, out_channels_conv1, out_channels_conv2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels_conv1, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels_conv1, out_channels_conv2, 3, padding=1),
        nn.ReLU(inplace=True)
    )


# Small UNet with skip connections to predict boat position from hits.
class BoatPredictionUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(3, 32, 32)
        self.dconv_down2 = double_conv(32, 64, 64)
        self.dconv_down3 = double_conv(64, 128, 128)

        self.conv_bottleneck = double_conv(128, 256, 128)

        self.dconv_up3 = double_conv(128 + 128, 128, 64)
        self.dconv_up2 = double_conv(64 + 64, 64, 32)
        self.dconv_up1 = double_conv(32 + 32, 32, 32)

        self.conv_last = nn.Conv2d(32, 3, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsampleSize2 = nn.Upsample(size=2)
        self.upsampleSize5 = nn.Upsample(size=5)
        self.upsampleSize10 = nn.Upsample(size=10)

    def forward(self, x):
        # Encoder
                                     #    3  x10x10
        conv1 = self.dconv_down1(x)  # -> 32 x10x10
        x = self.maxpool(conv1)      # -> 32 x5x5
        conv2 = self.dconv_down2(x)  # -> 64 x5x5
        x = self.maxpool(conv2)      # -> 64 x2x2
        conv3 = self.dconv_down3(x)  # -> 128x2x2
        x = self.maxpool(conv3)      # -> 128x1x1

        # Bottleneck
        self.conv_bottleneck(x)      # -> 128x1x1

        # Decoder
        x = self.upsampleSize2(x)        # -> 128x2x2
        x = torch.cat([x, conv3], dim=1) # -> 256x2x2
        x = self.dconv_up3(x) #          # -> 64 x2x2

        x = self.upsampleSize5(x)        # -> 64 x5x5
        x = torch.cat([x, conv2], dim=1) # -> 128x5x5
        x = self.dconv_up2(x)            # -> 32 x5x5

        x = self.upsampleSize10(x)       # -> 32 x10x10
        x = torch.cat([x, conv1], dim=1) # -> 64 x10x10
        x = self.dconv_up1(x)            # -> 32 x10x10

        out = self.conv_last(x)          # -> 3 x10x10

        return out
